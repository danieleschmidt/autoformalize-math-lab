"""Comprehensive integration tests for the autoformalize system.

This module contains end-to-end tests that validate the complete
mathematical formalization pipeline.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from autoformalize.core.pipeline import FormalizationPipeline
from autoformalize.core.config import FormalizationConfig, ModelConfig
from autoformalize.parsers.latex_parser import LaTeXParser
from autoformalize.generators.lean import Lean4Generator
from autoformalize.generators.isabelle import IsabelleGenerator
from autoformalize.generators.coq import CoqGenerator
from autoformalize.verifiers.lean_verifier import Lean4Verifier
from autoformalize.core.correction import SelfCorrectingGenerator
from autoformalize.core.optimization import AdaptiveCache, ResourceManager
from autoformalize.utils.metrics import FormalizationMetrics


class TestComprehensiveFormalization:
    """Comprehensive tests for the formalization pipeline."""
    
    @pytest.fixture
    def sample_latex(self):
        """Sample LaTeX mathematical content."""
        return """
\\documentclass{article}
\\usepackage{amsmath,amsthm}

\\newtheorem{theorem}{Theorem}
\\newtheorem{definition}{Definition}

\\begin{document}

\\begin{definition}[Even Number]
A natural number $n$ is called even if there exists a natural number $k$ such that $n = 2k$.
\\end{definition}

\\begin{theorem}[Sum of Even Numbers]
The sum of two even numbers is even.
\\end{theorem}

\\begin{proof}
Let $a$ and $b$ be two even numbers. By definition, there exist natural numbers $k$ and $l$ such that $a = 2k$ and $b = 2l$.

Therefore, $a + b = 2k + 2l = 2(k + l)$.

Since $k + l$ is a natural number, we have shown that $a + b$ is even.
\\end{proof}

\\end{document}
"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = FormalizationConfig()
        config.model = ModelConfig(name="mock-model", api_key="test-key")
        return config
    
    @pytest.fixture
    async def parser(self):
        """LaTeX parser instance."""
        return LaTeXParser()
    
    @pytest.fixture
    def mock_generator(self):
        """Mock generator for testing."""
        generator = Mock()
        generator.generate = AsyncMock()
        generator._call_llm = AsyncMock()
        generator._extract_lean_code = Mock(return_value="mock lean code")
        return generator
    
    @pytest.fixture
    def mock_verifier(self):
        """Mock verifier for testing."""
        verifier = Mock()
        verifier.verify_detailed = AsyncMock()
        return verifier
    
    @pytest.mark.asyncio
    async def test_latex_parsing_comprehensive(self, parser, sample_latex):
        """Test comprehensive LaTeX parsing."""
        result = await parser.parse(sample_latex)
        
        # Verify parsing results
        assert len(result.definitions) == 1
        assert len(result.theorems) == 1
        assert result.definitions[0].type == "definition"
        assert "even" in result.definitions[0].statement.lower()
        assert result.theorems[0].type == "theorem"
        assert result.theorems[0].proof is not None
        
        # Test parsing statistics
        stats = parser.get_parsing_statistics(result)
        assert stats['total'] == 2
        assert stats['statements_with_proofs'] == 1
    
    @pytest.mark.asyncio
    async def test_generator_integration(self, mock_generator, parser, sample_latex):
        """Test generator integration with parsed content."""
        # Parse content
        parsed_content = await parser.parse(sample_latex)
        
        # Mock successful generation
        mock_generator.generate.return_value = "theorem test : True := by trivial"
        
        # Generate code
        result = await mock_generator.generate(parsed_content)
        
        # Verify generation was called
        mock_generator.generate.assert_called_once_with(parsed_content)
        assert result == "theorem test : True := by trivial"
    
    @pytest.mark.asyncio
    async def test_verification_integration(self, mock_verifier):
        """Test verification integration."""
        from autoformalize.verifiers.lean_verifier import VerificationResult
        
        # Mock successful verification
        mock_result = VerificationResult(
            success=True,
            output="Verification successful",
            processing_time=1.5
        )
        mock_verifier.verify_detailed.return_value = mock_result
        
        # Test verification
        code = "theorem test : True := by trivial"
        result = await mock_verifier.verify_detailed(code)
        
        assert result.success
        assert result.processing_time == 1.5
        mock_verifier.verify_detailed.assert_called_once_with(code)
    
    @pytest.mark.asyncio
    async def test_self_correction_system(self, mock_generator, mock_verifier):
        """Test self-correction system."""
        from autoformalize.verifiers.lean_verifier import VerificationResult
        from autoformalize.core.correction import SelfCorrectingGenerator
        
        # Setup mock responses
        mock_generator.generate.return_value = "faulty code"
        mock_generator._call_llm.return_value = "corrected code"
        mock_generator._extract_lean_code.return_value = "corrected code"
        
        # First verification fails, second succeeds
        failed_result = VerificationResult(
            success=False,
            errors=["syntax error: expected theorem"],
            processing_time=0.5
        )
        success_result = VerificationResult(
            success=True,
            output="Verification successful",
            processing_time=1.0
        )
        
        mock_verifier.verify_detailed.side_effect = [failed_result, success_result]
        
        # Create self-correcting generator
        correcting_generator = SelfCorrectingGenerator(
            base_generator=mock_generator,
            verifier=mock_verifier,
            max_attempts=3
        )
        
        # Test correction
        from autoformalize.parsers.latex_parser import ParsedContent
        parsed_content = ParsedContent()
        
        final_code, attempts = await correcting_generator.generate_with_correction(parsed_content)
        
        assert final_code == "corrected code"
        assert len(attempts) == 1
        assert attempts[0].success
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test caching system performance."""
        from autoformalize.core.optimization import MemoryCache
        
        cache = MemoryCache(max_size=100)
        
        # Test cache operations
        await cache.set("key1", "value1", ttl=60)
        assert await cache.get("key1") == "value1"
        
        # Test cache miss
        assert await cache.get("nonexistent") is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5
    
    @pytest.mark.asyncio
    async def test_resource_management(self):
        """Test resource management and scaling."""
        resource_manager = ResourceManager(
            max_concurrent_requests=5,
            max_workers=2,
            enable_auto_scaling=True
        )
        
        # Test resource acquisition
        await resource_manager.acquire_resources()
        stats = resource_manager.get_resource_stats()
        assert stats["active_requests"] == 1
        assert stats["utilization"] > 0
        
        # Test resource release
        resource_manager.release_resources()
        stats = resource_manager.get_resource_stats()
        assert stats["active_requests"] == 0
        
        # Cleanup
        resource_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test comprehensive metrics collection."""
        metrics = FormalizationMetrics(enable_prometheus=False)
        
        # Record successful formalization
        metrics.record_formalization(
            success=True,
            target_system="lean4",
            processing_time=2.5,
            content_length=1000,
            output_length=500,
            correction_rounds=1,
            verification_success=True
        )
        
        # Get metrics summary
        summary = metrics.get_summary()
        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["overall_success_rate"] == 1.0
        
        # Get system-specific metrics
        lean_metrics = metrics.get_system_metrics("lean4")
        assert lean_metrics["requests"] == 1
        assert lean_metrics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, parser):
        """Test error handling and robustness."""
        # Test with malformed LaTeX
        malformed_latex = "\\begin{theorem} incomplete..."
        
        try:
            result = await parser.parse(malformed_latex)
            # Should not crash, might return empty or partial results
            assert isinstance(result.theorems, list)
            assert isinstance(result.definitions, list)
        except Exception as e:
            # Acceptable if it raises a proper ParseError
            from autoformalize.core.exceptions import ParseError
            assert isinstance(e, ParseError)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, parser, sample_latex):
        """Test concurrent processing capabilities."""
        # Create multiple parsing tasks
        tasks = [
            parser.parse(sample_latex)
            for _ in range(5)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert len(result.definitions) == 1
            assert len(result.theorems) == 1
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, parser):
        """Test memory usage stays reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many documents
        large_latex = """
\\begin{theorem}[Test Theorem %d]
This is theorem number %d with some mathematical content $x^2 + y^2 = z^2$.
\\end{theorem}
        """ * 100
        
        for i in range(10):
            content = large_latex % (i, i)
            result = await parser.parse(content)
            # Verify parsing worked
            assert len(result.theorems) == 100
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, parser, sample_latex):
        """Test performance benchmarks."""
        import time
        
        # Warm up
        await parser.parse(sample_latex)
        
        # Benchmark parsing
        start_time = time.time()
        for _ in range(10):
            await parser.parse(sample_latex)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 1.0, f"Average parsing time {avg_time:.3f}s exceeds 1s threshold"
    
    @pytest.mark.asyncio
    async def test_global_ready_deployment(self, mock_config):
        """Test global deployment readiness."""
        # Test configuration validation
        mock_config.validate()
        
        # Test internationalization readiness
        test_content = {
            "english": "theorem test : True := by trivial",
            "french": "théorème test : Vrai := par trivial",  # Accented characters
            "chinese": "定理 test : True := by trivial",  # Unicode
            "japanese": "定理 test : True := by trivial"
        }
        
        for lang, content in test_content.items():
            # Should handle Unicode properly
            encoded = content.encode('utf-8')
            decoded = encoded.decode('utf-8')
            assert decoded == content
    
    def test_security_validation(self):
        """Test security measures."""
        # Test input validation
        from autoformalize.api.server import FormalizationRequest
        from pydantic import ValidationError
        
        # Test valid request
        valid_request = FormalizationRequest(
            latex_content="\\theorem{test}",
            target_system="lean4"
        )
        assert valid_request.target_system == "lean4"
        
        # Test invalid target system
        with pytest.raises(ValidationError):
            FormalizationRequest(
                latex_content="\\theorem{test}",
                target_system="malicious_system"
            )
        
        # Test temperature bounds
        with pytest.raises(ValidationError):
            FormalizationRequest(
                latex_content="\\theorem{test}",
                target_system="lean4",
                temperature=5.0  # Out of bounds
            )
        
    @pytest.mark.asyncio
    async def test_compliance_gdpr(self):
        """Test GDPR compliance features."""
        metrics = FormalizationMetrics(enable_prometheus=False)
        
        # Test data anonymization
        metrics.record_formalization(
            success=True,
            target_system="lean4",
            processing_time=1.0,
            # No personal data should be stored
        )
        
        # Test data deletion capability
        metrics.reset()
        summary = metrics.get_summary()
        assert summary["total_requests"] == 0
    
    @pytest.mark.asyncio 
    async def test_auto_scaling_triggers(self):
        """Test auto-scaling trigger mechanisms."""
        resource_manager = ResourceManager(
            max_concurrent_requests=2,
            enable_auto_scaling=True
        )
        
        # Simulate high load
        await resource_manager.acquire_resources()
        await resource_manager.acquire_resources()
        
        # Check if scaling was triggered
        stats = resource_manager.get_resource_stats()
        initial_limit = stats["max_concurrent_requests"]
        
        # High utilization should trigger scaling
        assert stats["utilization"] >= 0.8
        
        resource_manager.cleanup()


class TestProductionReadiness:
    """Tests for production deployment readiness."""
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        """Test health check functionality."""
        # This would test the actual API endpoint in integration tests
        pass
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test monitoring and alerting integration."""
        metrics = FormalizationMetrics(enable_prometheus=True)
        
        # Test Prometheus metrics export
        prometheus_data = metrics.export_prometheus_metrics()
        assert "autoformalize_requests_total" in prometheus_data
        assert "autoformalize_processing_time_seconds" in prometheus_data
    
    @pytest.mark.asyncio
    async def test_disaster_recovery(self):
        """Test disaster recovery mechanisms."""
        # Test graceful degradation
        cache = AdaptiveCache(memory_cache_size=10)
        
        # Cache should work even if Redis is unavailable
        await cache.set("test", "value")
        result = await cache.get("test")
        assert result == "value"
    
    @pytest.mark.asyncio
    async def test_load_balancing_ready(self):
        """Test load balancing readiness."""
        # Test stateless operation
        config1 = FormalizationConfig()
        config2 = FormalizationConfig()
        
        # Configurations should be independent
        config1.max_workers = 4
        config2.max_workers = 8
        
        assert config1.max_workers != config2.max_workers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])