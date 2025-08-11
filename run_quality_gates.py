#!/usr/bin/env python3
"""Comprehensive Quality Gates Validation."""

import asyncio
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize import FormalizationPipeline


def run_security_scan() -> Dict[str, Any]:
    """Run security vulnerability scanning."""
    print("🔒 Running Security Scan...")
    
    try:
        # Run bandit security scanner
        result = subprocess.run([
            "bandit", "-r", "src/", "-f", "json"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ No high/medium security issues found")
            return {"status": "pass", "issues": 0}
        else:
            # Parse issues
            import json
            try:
                report = json.loads(result.stdout)
                issues = len(report.get("results", []))
                print(f"⚠️  Found {issues} potential security issues")
                return {"status": "warning", "issues": issues}
            except:
                print("⚠️  Security scan completed with warnings")
                return {"status": "warning", "issues": 1}
                
    except Exception as e:
        print(f"❌ Security scan failed: {e}")
        return {"status": "fail", "error": str(e)}


async def run_performance_benchmark() -> Dict[str, Any]:
    """Run performance benchmarking."""
    print("⚡ Running Performance Benchmark...")
    
    try:
        # Benchmark basic operations
        times = []
        
        for i in range(5):
            start = time.time()
            
            # Create pipeline (should be fast)
            pipeline = FormalizationPipeline(target_system="lean4")
            
            # Simple formalization
            latex = "\\begin{theorem}Test\\end{theorem}"
            await pipeline.formalize(latex, verify=False)
            
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"✅ Average processing time: {avg_time:.3f}s")
        print(f"✅ Maximum processing time: {max_time:.3f}s")
        
        # Performance criteria
        if avg_time < 0.5 and max_time < 1.0:
            return {"status": "pass", "avg_time": avg_time, "max_time": max_time}
        else:
            return {"status": "warning", "avg_time": avg_time, "max_time": max_time}
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return {"status": "fail", "error": str(e)}


async def run_functionality_tests() -> Dict[str, Any]:
    """Run core functionality tests."""
    print("🧪 Running Functionality Tests...")
    
    try:
        # Test basic functionality
        results = []
        
        # Test 1: Pipeline creation
        pipeline = FormalizationPipeline(target_system="lean4")
        results.append(pipeline is not None)
        
        # Test 2: LaTeX parsing
        latex = """
        \\begin{theorem}
        For any natural number $n$, we have $n + 0 = n$.
        \\end{theorem}
        """
        result = await pipeline.formalize(latex, verify=False)
        results.append(result.success)
        
        # Test 3: Multiple systems
        for system in ["lean4", "isabelle", "coq"]:
            pipe = FormalizationPipeline(target_system=system)
            result = await pipe.formalize(latex, verify=False)
            results.append(result.success)
        
        # Test 4: Error handling
        try:
            result = await pipeline.formalize("", verify=False)
            results.append(not result.success)  # Should fail gracefully
        except:
            results.append(False)
        
        passed = sum(results)
        total = len(results)
        
        print(f"✅ Functionality tests: {passed}/{total} passed")
        
        if passed == total:
            return {"status": "pass", "passed": passed, "total": total}
        elif passed >= total * 0.8:
            return {"status": "warning", "passed": passed, "total": total}
        else:
            return {"status": "fail", "passed": passed, "total": total}
            
    except Exception as e:
        print(f"❌ Functionality tests failed: {e}")
        return {"status": "fail", "error": str(e)}


def run_dependency_check() -> Dict[str, Any]:
    """Check critical dependencies."""
    print("📦 Checking Dependencies...")
    
    try:
        import autoformalize
        from autoformalize import FormalizationPipeline
        
        # Check version
        version = getattr(autoformalize, '__version__', 'unknown')
        print(f"✅ Package version: {version}")
        
        # Check core modules
        modules = [
            'autoformalize.core.pipeline',
            'autoformalize.parsers.latex_parser',
            'autoformalize.utils.logging_config'
        ]
        
        missing = []
        for module in modules:
            try:
                __import__(module)
            except ImportError as e:
                missing.append(module)
        
        if not missing:
            print("✅ All core modules available")
            return {"status": "pass", "missing": []}
        else:
            print(f"⚠️  Missing modules: {missing}")
            return {"status": "warning", "missing": missing}
            
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return {"status": "fail", "error": str(e)}


def generate_quality_report(results: Dict[str, Dict]) -> None:
    """Generate comprehensive quality report."""
    print("\n" + "="*60)
    print("📊 QUALITY GATES REPORT")
    print("="*60)
    
    overall_status = "PASS"
    
    for test_name, result in results.items():
        status = result.get("status", "unknown")
        print(f"\n{test_name.upper()}:")
        
        if status == "pass":
            print("  ✅ PASSED")
        elif status == "warning":
            print("  ⚠️  WARNING")
            if overall_status == "PASS":
                overall_status = "WARNING"
        else:
            print("  ❌ FAILED")
            overall_status = "FAIL"
        
        # Print details
        if "error" in result:
            print(f"    Error: {result['error']}")
        if "issues" in result:
            print(f"    Issues: {result['issues']}")
        if "passed" in result and "total" in result:
            print(f"    Tests: {result['passed']}/{result['total']}")
        if "avg_time" in result:
            print(f"    Avg Time: {result['avg_time']:.3f}s")
    
    print(f"\n{'='*60}")
    print(f"OVERALL STATUS: {overall_status}")
    print(f"{'='*60}\n")
    
    return overall_status == "PASS"


async def main():
    """Run all quality gates."""
    print("🚀 Starting Comprehensive Quality Gates")
    print("="*60)
    
    results = {}
    
    # Run all quality checks
    results["Security"] = run_security_scan()
    results["Performance"] = await run_performance_benchmark()  
    results["Functionality"] = await run_functionality_tests()
    results["Dependencies"] = run_dependency_check()
    
    # Generate report
    success = generate_quality_report(results)
    
    # Additional system info
    print("📋 SYSTEM INFORMATION:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Directory: {Path.cwd()}")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)