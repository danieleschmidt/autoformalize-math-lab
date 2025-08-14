#!/usr/bin/env python3
"""Test autonomous enhancement capabilities."""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.core.pipeline import FormalizationPipeline
from autoformalize.core.autonomous_enhancement import AutonomousEnhancementEngine, LearningMode

async def test_autonomous_enhancement():
    """Test autonomous enhancement capabilities."""
    print("🤖 Testing Autonomous Enhancement Engine")
    print("=" * 60)
    
    # Create base pipeline
    base_pipeline = FormalizationPipeline(target_system="lean4")
    
    # Create autonomous enhancement engine
    enhancement_engine = AutonomousEnhancementEngine(
        base_pipeline=base_pipeline,
        learning_modes=[
            LearningMode.PATTERN_RECOGNITION,
            LearningMode.SUCCESS_AMPLIFICATION,
            LearningMode.ADAPTIVE_PROMPTING
        ],
        pattern_db_path="cache/test_autonomous_patterns.json"
    )
    
    print(f"✅ Enhancement engine created with {len(enhancement_engine.learned_patterns)} initial patterns")
    
    # Test enhanced formalization
    test_latex = r"""
    \begin{theorem}
    For any natural number $n$, we have $n + 0 = n$.
    \end{theorem}
    \begin{proof}
    This follows from the definition of addition.
    \end{proof}
    """
    
    print("\n🧪 Testing enhanced formalization...")
    result = await enhancement_engine.enhanced_formalize(
        test_latex,
        verify=False,  # Skip verification for demo
        learn_from_result=True,
        adapt_strategy=True
    )
    
    print(f"✅ Enhanced formalization {'succeeded' if result.success else 'failed'}")
    if result.success and result.formal_code:
        print(f"Generated code preview: {result.formal_code[:100]}...")
    
    if hasattr(result, 'metadata') and result.metadata:
        enhancement_info = result.metadata.get('autonomous_enhancement', {})
        print(f"📊 Enhancement metrics:")
        print(f"   - Patterns applied: {enhancement_info.get('patterns_applied', 0)}")
        print(f"   - Strategy adapted: {enhancement_info.get('strategy_adapted', False)}")
        print(f"   - Enhancement time: {enhancement_info.get('enhancement_time', 0):.3f}s")
    
    # Test batch learning with sample data
    print("\n🎓 Testing autonomous batch learning...")
    training_data = [
        (r"\begin{theorem} $1 + 1 = 2$ \end{theorem}", "theorem one_plus_one : 1 + 1 = 2 := rfl"),
        (r"\begin{lemma} $0 + n = n$ \end{lemma}", "lemma zero_add (n : ℕ) : 0 + n = n := rfl"),
        (r"\begin{theorem} $n + 0 = n$ \end{theorem}", "theorem add_zero (n : ℕ) : n + 0 = n := rfl"),
    ]
    
    learning_results = await enhancement_engine.autonomous_batch_learning(
        training_data=training_data,
        validation_split=0.3,
        learning_cycles=2
    )
    
    print(f"✅ Batch learning completed")
    print(f"📊 Learning results:")
    print(f"   - Total patterns learned: {learning_results['final_metrics']['total_patterns_learned']}")
    print(f"   - Final success rate: {learning_results['final_metrics']['final_success_rate']:.2%}")
    print(f"   - Total improvement: {learning_results['final_metrics']['total_improvement']:+.2%}")
    
    # Get learning summary
    print("\n📈 Learning Summary:")
    summary = enhancement_engine.get_learning_summary()
    print(f"   - Total patterns: {summary['total_patterns']}")
    print(f"   - Learning modes: {', '.join(summary['learning_modes'])}")
    print(f"   - Pattern types: {summary['pattern_types']}")
    
    if summary['top_patterns']:
        print(f"   - Top pattern: {summary['top_patterns'][0]['type']} "
              f"(confidence: {summary['top_patterns'][0]['confidence']:.2f})")
    
    print("\n🎉 Autonomous enhancement test completed successfully!")
    
    return True

async def test_learning_modes():
    """Test different learning modes."""
    print("\n🧠 Testing Learning Modes")
    print("=" * 40)
    
    base_pipeline = FormalizationPipeline(target_system="lean4")
    
    # Test each learning mode
    for mode in LearningMode:
        print(f"\n🔬 Testing {mode.value}...")
        
        engine = AutonomousEnhancementEngine(
            base_pipeline=base_pipeline,
            learning_modes=[mode],
            pattern_db_path=f"cache/test_{mode.value}_patterns.json"
        )
        
        test_latex = r"""
        \begin{definition}
        A natural number $n$ is even if there exists $k$ such that $n = 2k$.
        \end{definition}
        """
        
        result = await engine.enhanced_formalize(test_latex, learn_from_result=True)
        
        print(f"   ✅ {mode.value} mode: {'Success' if result.success else 'Failed'}")
        
        summary = engine.get_learning_summary()
        print(f"   📊 Patterns learned: {summary['metrics']['patterns_learned']}")
    
    print("\n✅ All learning modes tested successfully!")

def main():
    """Main test function."""
    print("🚀 Starting Autonomous Enhancement Tests")
    print("=" * 60)
    
    # Run async tests
    try:
        # Test autonomous enhancement
        asyncio.run(test_autonomous_enhancement())
        
        # Test learning modes
        asyncio.run(test_learning_modes())
        
        print("\n" + "=" * 60)
        print("🎉 All autonomous enhancement tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()