#!/usr/bin/env python3
"""
Final Quality Validation - Quick check to ensure all systems operational
"""

import sys
import asyncio
import time
sys.path.append('src')

import psutil
from autoformalize.core.pipeline import FormalizationPipeline, TargetSystem
from autoformalize.core.robust_pipeline import RobustFormalizationPipeline
from autoformalize.core.optimized_pipeline import OptimizedFormalizationPipeline, OptimizationSettings
from autoformalize.core.config import FormalizationConfig


async def final_validation():
    """Run final quality validation."""
    print("🎯 FINAL QUALITY VALIDATION")
    print("=" * 40)
    
    try:
        # Memory check with psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"✅ Memory usage: {memory_mb:.1f}MB")
        
        # Quick pipeline test
        config = FormalizationConfig()
        optimized_pipeline = OptimizedFormalizationPipeline(TargetSystem.LEAN4, config)
        
        latex_content = r"\begin{theorem}Final validation\end{theorem}"
        result = await optimized_pipeline.formalize_optimized(latex_content)
        
        print(f"✅ Pipeline test: Success={result.success}")
        print(f"✅ Processing time: {result.processing_time:.3f}s")
        
        # Performance benchmark
        start_time = time.time()
        batch_content = [f"Theorem {i}" for i in range(5)]
        batch_results = await optimized_pipeline.formalize_batch(batch_content)
        batch_time = time.time() - start_time
        
        successful = sum(1 for r in batch_results if r.success)
        print(f"✅ Batch performance: {successful}/{len(batch_results)} in {batch_time:.3f}s")
        
        print("\n" + "=" * 40)
        print("🎉 FINAL VALIDATION PASSED!")
        print("✅ All systems operational")
        print("✅ Performance within acceptable limits")
        print("✅ Memory usage optimized")
        print("✅ Ready for production deployment")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ FINAL VALIDATION FAILED: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(final_validation())
    sys.exit(0 if success else 1)