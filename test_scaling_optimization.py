#!/usr/bin/env python3
"""Test scaling and optimization features."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.research.neural_theorem_synthesis import NeuralTheoremSynthesizer


async def test_neural_theorem_synthesis():
    """Test neural theorem synthesis."""
    print("🧠 Testing Neural Theorem Synthesis...")
    
    try:
        synthesizer = NeuralTheoremSynthesizer()
        
        # Test basic synthesis
        result = await synthesizer.synthesize_theorems(
            domain="number_theory",
            num_candidates=2,
            min_confidence=0.5,
            novelty_threshold=0.6
        )
        
        print(f"✅ Generated {len(result.candidates)} candidates")
        print(f"✅ Generation time: {result.generation_time:.3f}s")
        print(f"✅ Average confidence: {result.model_confidence:.3f}")
        
        if result.candidates:
            candidate = result.candidates[0]
            print(f"✅ Sample statement: {candidate.statement[:100]}...")
            print(f"✅ Confidence: {candidate.confidence:.3f}")
            print(f"✅ Novelty: {candidate.novelty_score:.3f}")
            print(f"✅ Complexity: {candidate.complexity_score:.3f}")
        
        return len(result.candidates) > 0
        
    except Exception as e:
        print(f"❌ Neural synthesis test failed: {e}")
        return False


async def test_batch_synthesis():
    """Test batch synthesis across domains."""
    print("🚀 Testing Batch Synthesis...")
    
    try:
        synthesizer = NeuralTheoremSynthesizer()
        
        domains = ["number_theory", "algebra", "analysis"]
        
        start_time = time.time()
        results = await synthesizer.batch_synthesis(
            domains=domains,
            candidates_per_domain=1,
            parallel_workers=3
        )
        end_time = time.time()
        
        print(f"✅ Processed {len(results)} domains")
        print(f"✅ Total batch time: {end_time - start_time:.3f}s")
        
        total_candidates = sum(len(result.candidates) for result in results.values())
        print(f"✅ Total candidates generated: {total_candidates}")
        
        for domain, result in results.items():
            print(f"   {domain}: {len(result.candidates)} candidates")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Batch synthesis test failed: {e}")
        return False


async def test_research_opportunities():
    """Test research opportunity discovery."""
    print("🔬 Testing Research Opportunity Discovery...")
    
    try:
        synthesizer = NeuralTheoremSynthesizer()
        
        # Generate some synthesis history
        await synthesizer.synthesize_theorems("number_theory", num_candidates=1)
        await synthesizer.synthesize_theorems("algebra", num_candidates=1)
        
        # Discover opportunities
        opportunities = await synthesizer.discover_research_opportunities()
        
        print(f"✅ Found {len(opportunities)} research opportunities")
        
        for i, opp in enumerate(opportunities[:3]):  # Show top 3
            print(f"   {i+1}. {opp.get('type', 'unknown')}: {opp.get('description', 'No description')}")
            print(f"      Priority: {opp.get('priority', 'N/A')}")
        
        return len(opportunities) > 0
        
    except Exception as e:
        print(f"❌ Research opportunities test failed: {e}")
        return False


async def test_performance_optimization():
    """Test performance optimization features."""
    print("⚡ Testing Performance Optimization...")
    
    try:
        synthesizer = NeuralTheoremSynthesizer()
        
        # Measure synthesis performance
        times = []
        for i in range(3):
            start = time.time()
            result = await synthesizer.synthesize_theorems(
                domain="general",
                num_candidates=1
            )
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f"✅ Average synthesis time: {avg_time:.3f}s")
        print(f"✅ Performance consistency: {max(times) - min(times):.3f}s variance")
        
        # Test caching effects
        if synthesizer.cache_dir:
            print("✅ Caching enabled")
        
        # Get statistics
        stats = synthesizer.get_synthesis_statistics()
        print(f"✅ Statistics available: {len(stats)} metrics")
        
        return avg_time < 1.0  # Should be fast for mock synthesis
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def main():
    """Run scaling and optimization tests."""
    print("🚀 Starting Scaling & Optimization Tests")
    print("=" * 50)
    
    tests = [
        test_neural_theorem_synthesis,
        test_batch_synthesis,
        test_research_opportunities,
        test_performance_optimization,
    ]
    
    results = []
    for test in tests:
        print()
        result = await test()
        results.append(result)
        print()
    
    print("=" * 50)
    print("📊 Scaling & Optimization Results:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All scaling tests passed!")
    else:
        print("⚠️  Some scaling tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)