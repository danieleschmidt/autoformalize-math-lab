"""Security tests for the formalization system."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from autoformalize.core.pipeline import FormalizationPipeline
from autoformalize.parsers.latex import LaTeXParser


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_latex_injection_protection(self):
        """Test protection against LaTeX code injection."""
        parser = LaTeXParser()
        
        # Test potentially dangerous LaTeX commands
        dangerous_inputs = [
            r"\write18{rm -rf /}",  # Shell escape
            r"\input{/etc/passwd}",  # File inclusion
            r"\immediate\write18{curl malicious.com}",  # Network access
            r"\openout\myfile=/tmp/malicious.txt",  # File writing
            r"\catcode`\{=12\catcode`\}=12",  # Catcode manipulation
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(Exception) or pytest.warns(UserWarning):
                parser.parse(dangerous_input)
    
    def test_file_path_traversal_protection(self, temp_dir):
        """Test protection against path traversal attacks."""
        pipeline = FormalizationPipeline()
        
        # Test directory traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "~/../../etc/hosts",
            "%SYSTEMROOT%\\system32\\config\\sam"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises(Exception):
                pipeline.process_file(Path(malicious_path))
    
    def test_oversized_input_handling(self):
        """Test handling of extremely large inputs."""
        parser = LaTeXParser()
        
        # Create very large input (10MB)
        large_input = "A" * (10 * 1024 * 1024)
        
        with pytest.raises(Exception) or pytest.warns(UserWarning):
            parser.parse(large_input)
    
    def test_malformed_unicode_handling(self):
        """Test handling of malformed Unicode input."""
        parser = LaTeXParser()
        
        # Test various malformed Unicode sequences
        malformed_inputs = [
            b'\xff\xfe\x00\x00'.decode('utf-8', errors='ignore'),  # BOM issues
            "Valid text \udcff invalid surrogate",  # Invalid surrogates
            "Text with null \x00 byte",  # Null bytes
            "\U00110000",  # Code point outside Unicode range
        ]
        
        for malformed_input in malformed_inputs:
            try:
                result = parser.parse(malformed_input)
                # If parsing succeeds, result should be safe
                assert result is not None
            except (UnicodeError, ValueError):
                # Expected to fail with malformed input
                pass
    
    def test_recursive_include_protection(self, temp_dir):
        """Test protection against recursive includes.""" 
        # Create files that include each other
        file_a = temp_dir / "a.tex"
        file_b = temp_dir / "b.tex"
        
        file_a.write_text(r"\input{b.tex}")
        file_b.write_text(r"\input{a.tex}")
        
        parser = LaTeXParser()
        
        with pytest.raises(Exception):  # Should detect recursion
            parser.parse_file(file_a)


@pytest.mark.security
class TestAPISecurity:
    """Test API security measures."""
    
    def test_api_key_handling(self, mock_llm_client):
        """Test secure handling of API keys."""
        # Ensure API keys are not logged or exposed
        with patch('autoformalize.core.pipeline.logger') as mock_logger:
            pipeline = FormalizationPipeline(
                llm_client=mock_llm_client,
                api_key="secret_api_key_12345"
            )
            
            mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
                "theorem secure : True := trivial"
            
            result = pipeline.formalize("\\begin{theorem}Test\\end{theorem}")
            
            # Check that API key is not in any log messages
            for call in mock_logger.info.call_args_list:
                assert "secret_api_key_12345" not in str(call)
            for call in mock_logger.debug.call_args_list:
                assert "secret_api_key_12345" not in str(call)
    
    def test_rate_limiting_protection(self, mock_llm_client):
        """Test rate limiting mechanisms."""
        from autoformalize.core.rate_limiter import RateLimiter
        
        # Create rate limiter (1 request per second)
        rate_limiter = RateLimiter(max_requests=1, time_window=1.0)
        
        pipeline = FormalizationPipeline(
            llm_client=mock_llm_client,
            rate_limiter=rate_limiter
        )
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem rate_limited : True := trivial"
        
        # First request should succeed
        result1 = pipeline.formalize("\\begin{theorem}Test 1\\end{theorem}")
        assert result1["success"] == True
        
        # Second immediate request should be rate limited
        with pytest.raises(Exception):  # Should raise rate limit error
            pipeline.formalize("\\begin{theorem}Test 2\\end{theorem}")
    
    def test_request_size_limits(self, mock_llm_client):
        """Test request size limiting."""
        pipeline = FormalizationPipeline(
            llm_client=mock_llm_client,
            max_request_size=1000  # 1KB limit
        )
        
        # Small request should work
        small_theorem = "\\begin{theorem}Small theorem\\end{theorem}"
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem small : True := trivial"
        
        result = pipeline.formalize(small_theorem)
        assert result["success"] == True
        
        # Large request should be rejected
        large_theorem = "\\begin{theorem}" + "A" * 2000 + "\\end{theorem}"
        
        with pytest.raises(Exception):
            pipeline.formalize(large_theorem)


@pytest.mark.security
class TestDataSanitization:
    """Test data sanitization and output safety."""
    
    def test_output_sanitization(self, mock_llm_client, mock_proof_assistant):
        """Test that generated output is sanitized."""
        pipeline = FormalizationPipeline(
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant
        )
        
        # Mock LLM response with potentially dangerous content
        dangerous_response = """
theorem test : True := 
  -- This is a comment with potential XSS <script>alert('xss')</script>
  trivial
        """
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            dangerous_response
        mock_proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        result = pipeline.formalize("\\begin{theorem}Test\\end{theorem}")
        
        # Check that dangerous content is sanitized or escaped
        assert "<script>" not in result["formal_proof"]
        assert "alert(" not in result["formal_proof"]
    
    def test_error_message_sanitization(self, mock_llm_client, mock_proof_assistant):
        """Test that error messages don't leak sensitive information."""
        pipeline = FormalizationPipeline(
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant
        )
        
        # Mock proof assistant error with sensitive info
        mock_proof_assistant.verify_proof.return_value = {
            "success": False,
            "errors": [
                "File not found: /home/user/.secrets/api_key.txt",
                "Database connection failed: postgresql://user:password@localhost/db"
            ],
            "warnings": []
        }
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem failing : False := sorry"
        
        result = pipeline.formalize("\\begin{theorem}Test\\end{theorem}")
        
        # Check that sensitive information is not in the result
        result_str = str(result)
        assert "password" not in result_str
        assert "api_key" not in result_str
        assert "/home/user/.secrets" not in result_str
    
    def test_log_sanitization(self, mock_llm_client):
        """Test that logs don't contain sensitive information."""
        with patch('autoformalize.core.pipeline.logger') as mock_logger:
            pipeline = FormalizationPipeline(
                llm_client=mock_llm_client,
                api_key="secret_key_123"
            )
            
            # Simulate an error that might log sensitive info
            mock_llm_client.chat.completions.create.side_effect = Exception(
                "API Error: Invalid key secret_key_123"
            )
            
            try:
                pipeline.formalize("\\begin{theorem}Test\\end{theorem}")
            except Exception:
                pass
            
            # Check that logs are sanitized
            all_log_calls = (
                mock_logger.error.call_args_list +
                mock_logger.warning.call_args_list +
                mock_logger.info.call_args_list +
                mock_logger.debug.call_args_list
            )
            
            for call in all_log_calls:
                call_str = str(call)
                assert "secret_key_123" not in call_str


@pytest.mark.security
class TestFileSystemSecurity:
    """Test file system security measures."""
    
    def test_temporary_file_security(self, temp_dir):
        """Test secure handling of temporary files."""
        pipeline = FormalizationPipeline()
        
        # Check that temporary files are created securely
        with pipeline._create_temp_file() as temp_file:
            # File should exist and be readable only by owner
            assert temp_file.exists()
            stat_info = temp_file.stat()
            
            # Check permissions (should be 600 or similar)
            import stat
            permissions = stat.filemode(stat_info.st_mode)
            assert "rw-------" in permissions or "rw-r--r--" in permissions
    
    def test_file_cleanup(self, temp_dir):
        """Test that temporary files are properly cleaned up."""
        pipeline = FormalizationPipeline()
        temp_files = []
        
        # Create several temporary files
        for i in range(5):
            with pipeline._create_temp_file() as temp_file:
                temp_file.write_text(f"temporary content {i}")
                temp_files.append(temp_file)
        
        # All files should be cleaned up after context exit
        for temp_file in temp_files:
            assert not temp_file.exists()
    
    def test_symlink_protection(self, temp_dir):
        """Test protection against symlink attacks."""
        pipeline = FormalizationPipeline()
        
        # Create a symlink to sensitive file
        sensitive_file = temp_dir / "sensitive.txt"
        sensitive_file.write_text("sensitive data")
        
        symlink = temp_dir / "symlink.tex"
        symlink.symlink_to(sensitive_file)
        
        # Pipeline should detect and reject symlinks
        with pytest.raises(Exception):
            pipeline.process_file(symlink)


@pytest.mark.security
class TestProofAssistantSecurity:
    """Test security of proof assistant interactions."""
    
    def test_proof_assistant_sandboxing(self, mock_proof_assistant):
        """Test that proof assistants run in sandboxed environment."""
        # This would test actual sandboxing in a real implementation
        pipeline = FormalizationPipeline(proof_assistant=mock_proof_assistant)
        
        # Mock proof that tries to execute system commands
        malicious_proof = """
theorem malicious : True := by
  sorry -- Actually tries to execute: system("rm -rf /")
"""
        
        mock_proof_assistant.verify_proof.return_value = {
            "success": False,
            "errors": ["Sandbox violation: system call blocked"],
            "warnings": []
        }
        
        result = pipeline.verify_proof(malicious_proof, "lean4")
        
        # Should fail due to sandbox protection
        assert result["success"] == False
        assert "sandbox" in result["errors"][0].lower()
    
    def test_resource_limits(self, mock_proof_assistant):
        """Test resource limits for proof assistant execution."""
        pipeline = FormalizationPipeline(
            proof_assistant=mock_proof_assistant,
            proof_timeout=1.0,  # 1 second timeout
            memory_limit=100 * 1024 * 1024  # 100MB limit
        )
        
        # Mock proof that would consume too many resources
        resource_heavy_proof = "theorem heavy : ∀ n : ℕ, complex_computation n := sorry"
        
        mock_proof_assistant.verify_proof.side_effect = TimeoutError("Proof timed out")
        
        with pytest.raises(TimeoutError):
            pipeline.verify_proof(resource_heavy_proof, "lean4")


@pytest.mark.security
class TestNetworkSecurity:
    """Test network security aspects."""
    
    @patch('requests.get')
    def test_url_validation(self, mock_get):
        """Test validation of URLs for arXiv fetching."""
        from autoformalize.parsers.arxiv import ArXivParser
        
        parser = ArXivParser()
        
        # Test malicious URLs
        malicious_urls = [
            "http://localhost:22/ssh",  # Local services
            "file:///etc/passwd",  # File protocol
            "ftp://malicious.com/exploit",  # Non-HTTP protocols
            "http://169.254.169.254/metadata",  # AWS metadata
        ]
        
        for url in malicious_urls:
            with pytest.raises(Exception):
                parser.fetch_url(url)
    
    @patch('requests.get')
    def test_response_size_limits(self, mock_get):
        """Test limits on response sizes from external services.""" 
        from autoformalize.parsers.arxiv import ArXivParser
        
        # Mock very large response
        mock_response = Mock()
        mock_response.headers = {'content-length': '100000000'}  # 100MB
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        parser = ArXivParser(max_response_size=1024*1024)  # 1MB limit
        
        with pytest.raises(Exception):
            parser.fetch_paper("2301.00001")
    
    def test_ssl_verification(self):
        """Test that SSL verification is enabled."""
        from autoformalize.parsers.arxiv import ArXivParser
        
        parser = ArXivParser()
        
        # Verify that SSL verification is not disabled
        assert parser.verify_ssl == True
        
        # Test with invalid SSL should fail
        with pytest.raises(Exception):
            parser.fetch_url("https://expired.badssl.com/")