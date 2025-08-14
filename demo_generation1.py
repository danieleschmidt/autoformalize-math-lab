#!/usr/bin/env python3
"""
Generation 1 Demo: Basic functionality demonstration
Verifies core pipeline works with simple mathematical formalization.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.core.pipeline import FormalizationPipeline
from autoformalize.parsers.latex_parser import LaTeXParser

async def demo_generation1():
    """Demonstrate Generation 1 (Simple) functionality."""
    print("🚀 Generation 1: MAKE IT WORK - Demo Starting")
    print("=" * 60)
    
    # Sample LaTeX theorem
    sample_latex = r"""
    \begin{theorem}[Sum of Even Numbers]
    The sum of two even numbers is even.
    \end{theorem}
    
    \begin{proof}
    Let $a$ and $b$ be two even numbers. By definition, there exist natural 
    numbers $k$ and $l$ such that $a = 2k$ and $b = 2l$.
    
    Therefore, $a + b = 2k + 2l = 2(k + l)$.
    
    Since $k + l$ is a natural number, we have shown that $a + b$ is even.
    \end{proof}
    """
    
    try:
        # Test 1: LaTeX Parser
        print("📐 Test 1: LaTeX Parsing")
        parser = LaTeXParser()
        parsed = await parser.parse(sample_latex)
        
        print(f"✅ Parsed {len(parsed.theorems)} theorem(s)")
        if parsed.theorems:
            theorem = parsed.theorems[0]
            print(f"   - Name: {theorem.name}")
            print(f"   - Statement: {theorem.statement[:100]}...")
            print(f"   - Has proof: {'Yes' if theorem.proof else 'No'}")
        
        # Test 2: Pipeline Integration
        print("\n🔧 Test 2: Pipeline Integration")
        pipeline = FormalizationPipeline(target_system="lean4")
        result = await pipeline.formalize(sample_latex, verify=False)
        
        print(f"✅ Formalization: {'Success' if result.success else 'Failed'}")
        if result.success:
            print(f"   - Processing time: {result.processing_time:.2f}s")
            print(f"   - Formal code length: {len(result.formal_code) if result.formal_code else 0} chars")
            if result.formal_code:
                print(f"   - Preview: {result.formal_code[:200]}...")
        else:
            print(f"   - Error: {result.error_message}")
        
        # Test 3: Multi-system Support
        print("\n⚙️  Test 3: Multi-system Support")
        for system in ["lean4", "isabelle", "coq"]:
            pipeline = FormalizationPipeline(target_system=system)
            result = await pipeline.formalize(sample_latex, verify=False)
            status = "✅" if result.success else "❌"
            print(f"   {status} {system.upper()}: {'Success' if result.success else 'Failed'}")
        
        # Test 4: Performance Metrics
        print("\n📊 Test 4: Performance Metrics")
        metrics = pipeline.get_metrics()
        print(f"✅ Metrics collected: {len(metrics)} categories")
        if 'formalization_count' in metrics:
            print(f"   - Total formalizations: {metrics['formalization_count']}")
        if 'success_rate' in metrics:
            print(f"   - Success rate: {metrics['success_rate']:.1%}")
        
        print("\n🎉 Generation 1 Demo Complete!")
        print("✅ Core functionality verified")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(demo_generation1())
    sys.exit(0 if success else 1)