#!/usr/bin/env python3
"""Test robust pipeline functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize import FormalizationPipeline


async def test_offline_pipeline():
    """Test pipeline works without API keys."""
    print("🧪 Testing Offline Pipeline...")
    
    try:
        # Create pipeline without API key - should use mock components
        pipeline = FormalizationPipeline(
            target_system="lean4",
            model="gpt-4"
        )
        
        print(f"✅ Pipeline created: {type(pipeline.generator).__name__}")
        
        # Test formalization
        latex_content = """
        \\begin{theorem}
        For any natural number $n$, we have $n + 0 = n$.
        \\end{theorem}
        """
        
        result = await pipeline.formalize(latex_content, verify=False)
        
        print(f"✅ Formalization {'succeeded' if result.success else 'failed'}")
        if result.success:
            print(f"✅ Generated code: {result.formal_code[:50]}...")
        else:
            print(f"❌ Error: {result.error_message}")
            
        return result.success
        
    except Exception as e:
        print(f"❌ Offline pipeline test failed: {e}")
        return False


async def test_error_recovery():
    """Test pipeline error recovery."""
    print("🧪 Testing Error Recovery...")
    
    try:
        pipeline = FormalizationPipeline(target_system="lean4")
        
        # Test with invalid LaTeX
        invalid_latex = "\\invalid{broken latex content}"
        
        result = await pipeline.formalize(invalid_latex, verify=False)
        
        # Should handle gracefully
        print(f"✅ Invalid input handled: success={result.success}")
        
        # Test with empty content
        result2 = await pipeline.formalize("", verify=False)
        print(f"✅ Empty input handled: success={result2.success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        return False


async def test_multiple_systems():
    """Test multiple target systems."""
    print("🧪 Testing Multiple Target Systems...")
    
    systems = ["lean4", "isabelle", "coq"]
    results = []
    
    for system in systems:
        try:
            pipeline = FormalizationPipeline(target_system=system)
            
            latex_content = "\\begin{theorem}Test theorem\\end{theorem}"
            result = await pipeline.formalize(latex_content, verify=False)
            
            print(f"✅ {system}: {'success' if result.success else 'failed'}")
            results.append(result.success)
            
        except Exception as e:
            print(f"❌ {system} failed: {e}")
            results.append(False)
    
    return all(results)


async def main():
    """Run robustness tests."""
    print("🛡️ Starting Robustness Tests")
    print("=" * 50)
    
    tests = [
        test_offline_pipeline,
        test_error_recovery,
        test_multiple_systems,
    ]
    
    results = []
    for test in tests:
        print()
        result = await test()
        results.append(result)
        print()
    
    print("=" * 50)
    print("📊 Robustness Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All robustness tests passed!")
    else:
        print("⚠️  Some robustness tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)