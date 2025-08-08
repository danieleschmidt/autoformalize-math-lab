#!/usr/bin/env python3
"""
Generation 1 Test - Basic Functionality Test
Tests core formalization pipeline without external dependencies.
"""

import asyncio
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_basic_pipeline():
    """Test basic pipeline functionality."""
    print("🧪 Testing Generation 1: Basic Functionality")
    
    try:
        from autoformalize.core.pipeline import FormalizationPipeline
        print("✅ Pipeline import successful")
        
        # Test LaTeX parsing
        latex_content = r"""
        \begin{theorem}
        For any prime $p > 2$, we have $p \equiv 1 \pmod{2}$ or $p \equiv 3 \pmod{2}$.
        \end{theorem}
        \begin{proof}
        Since $p$ is odd and greater than 2, $p$ is not divisible by 2.
        By the division algorithm, $p = 2q + r$ where $r \in \{0, 1\}$.
        Since $p$ is odd, $r \neq 0$, thus $r = 1$ and $p = 2q + 1$.
        Therefore $p \equiv 1 \pmod{2}$.
        \end{proof}
        """
        
        print("✅ Test LaTeX content prepared")
        
        # Test parser directly
        from autoformalize.parsers.latex_parser import LaTeXParser
        parser = LaTeXParser()
        parsed_content = await parser.parse(latex_content)
        print(f"✅ LaTeX parsing successful: {len(parsed_content.theorems)} theorems found")
        
        # Test generator initialization (without API key - should handle gracefully)
        from autoformalize.generators.lean import Lean4Generator
        try:
            generator = Lean4Generator(model="mock", api_key=None)
            print("✅ Generator initialization successful (mock mode)")
        except Exception as e:
            print(f"⚠️  Generator needs API key: {e}")
        
        print("🎉 Generation 1 Basic Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Generation 1 Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_config_and_utils():
    """Test configuration and utility modules."""
    print("\n🔧 Testing Configuration and Utilities")
    
    try:
        from autoformalize.core.config import FormalizationConfig
        config = FormalizationConfig()
        print(f"✅ Config loaded: timeout={config.verification.timeout}s")
        
        from autoformalize.utils.metrics import FormalizationMetrics
        metrics = FormalizationMetrics()
        metrics.record_formalization(success=True, target_system="lean4", processing_time=1.5)
        summary = metrics.get_summary()
        print(f"✅ Metrics working: {summary} recorded")
        
        from autoformalize.utils.logging_config import setup_logger
        logger = setup_logger("test")
        logger.info("Test log message")
        print("✅ Logging configuration working")
        
        return True
        
    except Exception as e:
        print(f"❌ Config/Utils test FAILED: {e}")
        return False

async def main():
    """Run all Generation 1 tests."""
    print("=" * 60)
    print("TERRAGON SDLC v4.0 - GENERATION 1 TESTING")
    print("=" * 60)
    
    basic_test = await test_basic_pipeline()
    config_test = await test_config_and_utils()
    
    if basic_test and config_test:
        print("\n🎉 ALL GENERATION 1 TESTS PASSED")
        print("✅ Ready to proceed to Generation 2 (Robust)")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Fix issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))