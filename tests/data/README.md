# Test Data

This directory contains test data files used by the test suite.

## Structure

- `latex/` - Sample LaTeX files for testing parsers
- `proofs/` - Sample formal proofs for testing generators
- `benchmarks/` - Small benchmark datasets for testing evaluation
- `fixtures/` - Static test fixtures and expected outputs

## Usage

Test data is accessed via the `test_data_dir` fixture in `conftest.py`:

```python
def test_example(test_data_dir):
    latex_file = test_data_dir / "latex" / "sample_theorem.tex"
    # Use the test file...
```

## Adding New Test Data

When adding new test data:

1. Place files in appropriate subdirectories
2. Keep files small (< 1MB) to avoid repository bloat
3. Include a README in subdirectories explaining the data
4. Use descriptive filenames that indicate the test purpose
5. Avoid including large datasets - use synthetic data instead

## Copyright Notice

Test data files may contain mathematical content from various sources.
All content is used for testing purposes only under fair use provisions.
Original sources are credited where applicable.