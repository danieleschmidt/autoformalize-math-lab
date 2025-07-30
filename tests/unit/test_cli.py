"""Unit tests for CLI module."""

import pytest
from click.testing import CliRunner
import tempfile
from pathlib import Path

from autoformalize.cli import main


class TestCLI:
    """Test cases for the CLI interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_help(self):
        """Test main command help."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'autoformalize-math-lab' in result.output
        assert 'Automated Mathematical Formalization' in result.output

    def test_version_option(self):
        """Test version option."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output

    def test_debug_flag(self):
        """Test debug flag."""
        result = self.runner.invoke(main, ['--debug', '--help'])
        assert result.exit_code == 0
        assert 'Debug mode enabled' in result.output

    def test_formalize_help(self):
        """Test formalize command help."""
        result = self.runner.invoke(main, ['formalize', '--help'])
        assert result.exit_code == 0
        assert 'Formalize a single LaTeX file' in result.output

    def test_formalize_with_nonexistent_file(self):
        """Test formalize command with non-existent file."""
        result = self.runner.invoke(main, ['formalize', 'nonexistent.tex'])
        assert result.exit_code != 0

    def test_formalize_with_valid_file(self):
        """Test formalize command with valid file."""
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as tmp:
            tmp.write(b'\\begin{theorem}Test\\end{theorem}')
            tmp.flush()
            
            result = self.runner.invoke(main, ['formalize', tmp.name])
            assert result.exit_code == 0
            assert 'Formalizing' in result.output
            assert 'Implementation coming soon' in result.output
            
            Path(tmp.name).unlink()  # Clean up

    def test_formalize_target_options(self):
        """Test formalize command with different target options."""
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as tmp:
            tmp.write(b'\\begin{theorem}Test\\end{theorem}')
            tmp.flush()
            
            for target in ['lean4', 'isabelle', 'coq', 'agda']:
                result = self.runner.invoke(main, ['formalize', tmp.name, '--target', target])
                assert result.exit_code == 0
                assert f'Target: {target}' in result.output
            
            Path(tmp.name).unlink()

    def test_batch_help(self):
        """Test batch command help."""
        result = self.runner.invoke(main, ['batch', '--help'])
        assert result.exit_code == 0
        assert 'Batch process multiple LaTeX files' in result.output

    def test_batch_with_nonexistent_dir(self):
        """Test batch command with non-existent directory."""
        result = self.runner.invoke(main, ['batch', 'nonexistent_dir'])
        assert result.exit_code != 0

    def test_arxiv_help(self):
        """Test arxiv command help."""
        result = self.runner.invoke(main, ['arxiv', '--help'])
        assert result.exit_code == 0
        assert 'Process an arXiv paper' in result.output

    def test_arxiv_command(self):
        """Test arxiv command."""
        result = self.runner.invoke(main, ['arxiv', '2301.00001'])
        assert result.exit_code == 0
        assert 'Processing arXiv paper: 2301.00001' in result.output

    def test_evaluate_help(self):
        """Test evaluate command help."""
        result = self.runner.invoke(main, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert 'Run evaluation on benchmark datasets' in result.output

    def test_evaluate_command(self):
        """Test evaluate command."""
        result = self.runner.invoke(main, ['evaluate'])
        assert result.exit_code == 0
        assert 'Running evaluation' in result.output
        assert 'undergraduate_math' in result.output

    def test_scoreboard_help(self):
        """Test scoreboard command help."""
        result = self.runner.invoke(main, ['scoreboard', '--help'])
        assert result.exit_code == 0
        assert 'Display or update the formalization success scoreboard' in result.output

    def test_validate_help(self):
        """Test validate command help."""
        result = self.runner.invoke(main, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'Validate generated formal proofs' in result.output

    def test_validate_no_files(self):
        """Test validate command with no files."""
        result = self.runner.invoke(main, ['validate'])
        assert result.exit_code == 0
        assert 'No files specified' in result.output

    def test_interactive_help(self):
        """Test interactive command help."""
        result = self.runner.invoke(main, ['interactive', '--help'])
        assert result.exit_code == 0
        assert 'Start interactive formalization session' in result.output

    def test_doctor_help(self):
        """Test doctor command help."""
        result = self.runner.invoke(main, ['doctor', '--help'])
        assert result.exit_code == 0
        assert 'Run diagnostic checks' in result.output

    def test_doctor_command(self):
        """Test doctor command."""
        result = self.runner.invoke(main, ['doctor'])
        assert result.exit_code == 0
        assert 'Running diagnostic checks' in result.output
        assert 'Python dependencies' in result.output

    @pytest.mark.parametrize("target", ["lean4", "isabelle", "coq", "agda"])
    def test_output_file_extensions(self, target):
        """Test that output files get correct extensions."""
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as tmp:
            tmp.write(b'\\begin{theorem}Test\\end{theorem}')
            tmp.flush()
            
            result = self.runner.invoke(main, ['formalize', tmp.name, '--target', target])
            assert result.exit_code == 0
            
            expected_extensions = {
                'lean4': '.lean',
                'isabelle': '.thy', 
                'coq': '.v',
                'agda': '.agda'
            }
            
            assert expected_extensions[target] in result.output
            Path(tmp.name).unlink()