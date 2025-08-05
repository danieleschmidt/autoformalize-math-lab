#!/usr/bin/env python3
"""Basic functionality test for the autoformalize system."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize import FormalizationPipeline, LaTeXParser


async def test_latex_parser():
    """Test LaTeX parser functionality."""
    print("🧪 Testing LaTeX Parser...")
    
    parser = LaTeXParser()
    
    latex_content = """
    \\begin{theorem}[Test Theorem]
    For any natural number $n$, we have $n + 0 = n$.
    \\end{theorem}
    \\begin{proof}
    This follows from the definition of addition.
    \\end{proof}
    """
    
    try:
        result = await parser.parse(latex_content)
        
        print(f"✅ Parsed {len(result.theorems)} theorems")
        print(f"✅ Parsed {len(result.definitions)} definitions")
        print(f"✅ Extracted {len(result.raw_math)} math expressions")
        
        if result.theorems:
            theorem = result.theorems[0]
            print(f"✅ Theorem name: {theorem.name}")
            print(f"✅ Theorem statement: {theorem.statement[:50]}...")
            print(f"✅ Has proof: {'Yes' if theorem.proof else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ LaTeX parser test failed: {e}")
        return False


async def test_pipeline_creation():
    """Test pipeline creation."""
    print("🧪 Testing Pipeline Creation...")
    
    try:
        pipeline = FormalizationPipeline(
            target_system="lean4",
            model="gpt-4"
        )
        
        print(f"✅ Pipeline created for {pipeline.target_system.value}")
        print(f"✅ Using model: {pipeline.model}")
        print(f"✅ Parser: {type(pipeline.parser).__name__}")
        print(f"✅ Generator: {type(pipeline.generator).__name__}")
        print(f"✅ Verifier: {type(pipeline.verifier).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline creation test failed: {e}")
        return False


async def test_file_parsing():
    """Test parsing from file."""
    print("🧪 Testing File Parsing...")
    
    parser = LaTeXParser()
    test_file = Path("examples/simple_theorem.tex")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        result = await parser.parse_file(test_file)
        
        print(f"✅ File parsed successfully")
        print(f"✅ Found {len(result.theorems)} theorems")
        print(f"✅ Found {len(result.definitions)} definitions")
        print(f"✅ Found {len(result.lemmas)} lemmas")
        
        # Print details
        for i, theorem in enumerate(result.theorems):
            print(f"   Theorem {i+1}: {theorem.name or 'Unnamed'}")
        
        for i, definition in enumerate(result.definitions):
            print(f"   Definition {i+1}: {definition.name or 'Unnamed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ File parsing test failed: {e}")
        return False


async def main():
    """Run all basic functionality tests."""
    print("🚀 Starting Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_latex_parser,
        test_pipeline_creation,
        test_file_parsing,
    ]
    
    results = []
    for test in tests:
        print()
        result = await test()
        results.append(result)
        print()
    
    print("=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)