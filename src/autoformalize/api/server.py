"""FastAPI web server for the autoformalize service.

This module provides a REST API and web interface for the
mathematical formalization pipeline.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..core.pipeline import FormalizationPipeline
from ..core.config import FormalizationConfig
from ..core.optimization import AdaptiveCache, ResourceManager
from ..utils.metrics import FormalizationMetrics
from ..utils.logging_config import setup_logger


# Pydantic models for API
class FormalizationRequest(BaseModel):
    """Request model for formalization."""
    latex_content: str = Field(..., description="LaTeX mathematical content")
    target_system: str = Field(..., description="Target proof assistant system", regex="^(lean4|isabelle|coq|agda)$")
    model: Optional[str] = Field("gpt-4", description="LLM model to use")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Generation temperature")
    max_correction_attempts: Optional[int] = Field(3, ge=0, le=10, description="Maximum correction attempts")
    enable_verification: Optional[bool] = Field(True, description="Enable proof verification")


class FormalizationResponse(BaseModel):
    """Response model for formalization."""
    request_id: str
    success: bool
    target_system: str
    generated_code: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    correction_attempts: List[Dict[str, Any]] = []
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class StatusResponse(BaseModel):
    """Response model for status check."""
    status: str
    version: str
    active_requests: int
    total_processed: int
    success_rate: float
    cache_hit_rate: float
    system_resources: Dict[str, Any]
    proof_assistant_status: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    summary: Dict[str, Any]
    by_system: Dict[str, Any]
    error_analysis: Dict[str, Any]
    cache_performance: Dict[str, Any]
    resource_utilization: Dict[str, Any]


# Global state
app_state = {
    "pipeline": None,
    "config": None,
    "cache": None,
    "resource_manager": None,
    "metrics": None,
    "active_requests": {},
    "logger": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = setup_logger(__name__)
    app_state["logger"] = logger
    
    logger.info("Starting autoformalize API server...")
    
    try:
        # Initialize configuration
        config = FormalizationConfig()
        app_state["config"] = config
        
        # Initialize cache
        cache = AdaptiveCache(
            memory_cache_size=1000,
            redis_url=None,  # Configure as needed
            enable_adaptive=True
        )
        app_state["cache"] = cache
        
        # Initialize resource manager
        resource_manager = ResourceManager(
            max_concurrent_requests=config.max_workers * 2,
            max_workers=config.max_workers,
            enable_auto_scaling=True
        )
        app_state["resource_manager"] = resource_manager
        
        # Initialize metrics
        metrics = FormalizationMetrics(enable_prometheus=True)
        app_state["metrics"] = metrics
        
        # Initialize pipeline
        pipeline = FormalizationPipeline(config)
        await pipeline.initialize()
        app_state["pipeline"] = pipeline
        
        logger.info("Autoformalize API server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down autoformalize API server...")
    
    if app_state["resource_manager"]:
        app_state["resource_manager"].cleanup()
    
    if app_state["cache"]:
        await app_state["cache"].clear()
    
    logger.info("Autoformalize API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Autoformalize API",
    description="Mathematical formalization service using LLMs and formal proof assistants",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get current state
def get_pipeline() -> FormalizationPipeline:
    """Get the current pipeline instance."""
    if not app_state["pipeline"]:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return app_state["pipeline"]


def get_metrics() -> FormalizationMetrics:
    """Get the current metrics instance."""
    if not app_state["metrics"]:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    return app_state["metrics"]


def get_cache() -> AdaptiveCache:
    """Get the current cache instance."""
    if not app_state["cache"]:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    return app_state["cache"]


def get_resource_manager() -> ResourceManager:
    """Get the current resource manager."""
    if not app_state["resource_manager"]:
        raise HTTPException(status_code=503, detail="Resource manager not initialized")
    return app_state["resource_manager"]


# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Autoformalize - Mathematical Formalization Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            textarea, select, input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            textarea { height: 200px; font-family: monospace; }
            button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #2980b9; }
            .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 4px; }
            .success { border-left: 4px solid #27ae60; }
            .error { border-left: 4px solid #e74c3c; }
            pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Autoformalize - Mathematical Formalization Service</h1>
            
            <div class="section">
                <h2>Convert LaTeX Mathematics to Formal Proofs</h2>
                <p>Transform your mathematical content into verified formal proofs using state-of-the-art LLMs and proof assistants.</p>
                
                <div class="form-group">
                    <label for="latex">LaTeX Mathematical Content:</label>
                    <textarea id="latex" placeholder="Enter your LaTeX mathematical content here...">\\begin{theorem}[Pythagorean Theorem]
For a right triangle with legs of length $a$ and $b$, and hypotenuse of length $c$, we have $a^2 + b^2 = c^2$.
\\end{theorem}

\\begin{proof}
Consider a square with side length $(a + b)$...
\\end{proof}</textarea>
                </div>
                
                <div class="form-group">
                    <label for="target">Target Proof Assistant:</label>
                    <select id="target">
                        <option value="lean4">Lean 4</option>
                        <option value="isabelle">Isabelle/HOL</option>  
                        <option value="coq">Coq</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="model">LLM Model:</label>
                    <select id="model">
                        <option value="gpt-4">GPT-4</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                    </select>
                </div>
                
                <button onclick="formalize()">🚀 Formalize</button>
                <button onclick="clearResult()">🗑️ Clear</button>
            </div>
            
            <div id="result" class="result" style="display:none;"></div>
        </div>
        
        <script>
        async function formalize() {
            const latex = document.getElementById('latex').value;
            const target = document.getElementById('target').value;
            const model = document.getElementById('model').value;
            const resultDiv = document.getElementById('result');
            
            if (!latex.trim()) {
                alert('Please enter LaTeX content');
                return;
            }
            
            // Show loading
            resultDiv.innerHTML = '<p>🔄 Processing your mathematical content...</p>';
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';
            
            try {
                const response = await fetch('/api/v1/formalize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        latex_content: latex,
                        target_system: target,
                        model: model,
                        enable_verification: true
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <h3>✅ Formalization Successful!</h3>
                        <p><strong>Target System:</strong> ${result.target_system.toUpperCase()}</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>
                        <p><strong>Correction Attempts:</strong> ${result.correction_attempts.length}</p>
                        
                        <h4>Generated Code:</h4>
                        <pre>${result.generated_code}</pre>
                        
                        ${result.verification_result ? `
                        <h4>Verification Result:</h4>
                        <p><strong>Status:</strong> ${result.verification_result.success ? '✅ Verified' : '❌ Failed'}</p>
                        ` : ''}
                    `;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `
                        <h3>❌ Formalization Failed</h3>
                        <p><strong>Error:</strong> ${result.error_message}</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>
                    `;
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `
                    <h3>❌ Request Failed</h3>
                    <p><strong>Error:</strong> ${error.message}</p>
                `;
            }
        }
        
        function clearResult() {
            document.getElementById('result').style.display = 'none';
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/v1/formalize", response_model=FormalizationResponse)
async def formalize(
    request: FormalizationRequest,
    background_tasks: BackgroundTasks,
    pipeline: FormalizationPipeline = Depends(get_pipeline),
    metrics: FormalizationMetrics = Depends(get_metrics),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Formalize mathematical content."""
    
    # Generate request ID
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Acquire resources
    await resource_manager.acquire_resources()
    
    try:
        start_time = time.time()
        
        # Track processing
        processing_metrics = metrics.start_processing(
            target_system=request.target_system,
            content_length=len(request.latex_content)
        )
        
        # Store active request
        app_state["active_requests"][request_id] = {
            "start_time": start_time,
            "target_system": request.target_system,
            "status": "processing"
        }
        
        try:
            # Run formalization
            result = await pipeline.formalize(
                latex_content=request.latex_content,
                target_system=request.target_system,
                model=request.model,
                temperature=request.temperature,
                max_correction_attempts=request.max_correction_attempts,
                enable_verification=request.enable_verification
            )
            
            processing_time = time.time() - start_time
            
            # Record metrics
            metrics.record_formalization(
                success=result.success,
                target_system=request.target_system,
                processing_time=processing_time,
                content_length=len(request.latex_content),
                output_length=len(result.generated_code) if result.generated_code else 0,
                correction_rounds=len(result.correction_attempts),
                verification_success=result.verification_result.get("success") if result.verification_result else None
            )
            
            # Prepare response
            response = FormalizationResponse(
                request_id=request_id,
                success=result.success,
                target_system=request.target_system,
                generated_code=result.generated_code,
                verification_result=result.verification_result,
                correction_attempts=[attempt.to_dict() for attempt in result.correction_attempts] if hasattr(result, 'correction_attempts') else [],
                processing_time=processing_time,
                error_message=result.error_message,
                metadata={
                    "model_used": request.model,
                    "temperature": request.temperature,
                    "verification_enabled": request.enable_verification
                }
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record failed attempt
            metrics.record_formalization(
                success=False,
                target_system=request.target_system,
                processing_time=processing_time,
                content_length=len(request.latex_content),
                error=str(e)
            )
            
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up active request
            if request_id in app_state["active_requests"]:
                del app_state["active_requests"][request_id]
    
    finally:
        # Release resources
        resource_manager.release_resources()


@app.post("/api/v1/formalize/file")
async def formalize_file(
    file: UploadFile = File(...),
    target_system: str = "lean4",
    model: str = "gpt-4",
    pipeline: FormalizationPipeline = Depends(get_pipeline)
):
    """Formalize mathematical content from uploaded file."""
    
    if not file.filename.endswith(('.tex', '.txt')):
        raise HTTPException(status_code=400, detail="Only .tex and .txt files are supported")
    
    try:
        # Read file content
        content = await file.read()
        latex_content = content.decode('utf-8')
        
        # Create formalization request
        request = FormalizationRequest(
            latex_content=latex_content,
            target_system=target_system,
            model=model
        )
        
        # Process request
        return await formalize(request, BackgroundTasks(), pipeline, get_metrics(), get_resource_manager())
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status(
    metrics: FormalizationMetrics = Depends(get_metrics),
    resource_manager: ResourceManager = Depends(get_resource_manager),
    pipeline: FormalizationPipeline = Depends(get_pipeline)
):
    """Get service status and health information."""
    
    # Get metrics summary
    metrics_summary = metrics.get_summary()
    
    # Get resource stats
    resource_stats = resource_manager.get_resource_stats()
    
    # Check proof assistant installations
    proof_assistant_status = {}
    
    # This would check actual installations in production
    proof_assistant_status = {
        "lean4": {"installed": False, "version": "Not available"},
        "isabelle": {"installed": False, "version": "Not available"},
        "coq": {"installed": False, "version": "Not available"}
    }
    
    return StatusResponse(
        status="healthy",
        version="1.0.0",
        active_requests=len(app_state["active_requests"]),
        total_processed=metrics_summary.get("total_requests", 0),
        success_rate=metrics_summary.get("overall_success_rate", 0.0),
        cache_hit_rate=0.0,  # Would get from cache
        system_resources=resource_stats,
        proof_assistant_status=proof_assistant_status
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics_endpoint(
    metrics: FormalizationMetrics = Depends(get_metrics),
    cache: AdaptiveCache = Depends(get_cache),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Get detailed metrics and analytics."""
    
    return MetricsResponse(
        summary=metrics.get_summary(),
        by_system={
            system: metrics.get_system_metrics(system)
            for system in ["lean4", "isabelle", "coq"]
        },
        error_analysis=metrics.get_error_analysis(),
        cache_performance=cache.get_performance_stats(),
        resource_utilization=resource_manager.get_resource_stats()
    )


@app.get("/api/v1/metrics/prometheus")
async def get_prometheus_metrics(metrics: FormalizationMetrics = Depends(get_metrics)):
    """Get metrics in Prometheus format."""
    prometheus_data = metrics.export_prometheus_metrics()
    return Response(content=prometheus_data, media_type="text/plain")


@app.delete("/api/v1/cache")
async def clear_cache(cache: AdaptiveCache = Depends(get_cache)):
    """Clear all cached data."""
    await cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/api/v1/systems")
async def get_supported_systems():
    """Get list of supported proof assistant systems."""
    return {
        "systems": [
            {
                "id": "lean4",
                "name": "Lean 4",
                "description": "Modern theorem prover with dependent types",
                "website": "https://lean-lang.org/"
            },
            {
                "id": "isabelle",
                "name": "Isabelle/HOL",
                "description": "Generic proof assistant for Higher-Order Logic",
                "website": "https://isabelle.in.tum.de/"
            },
            {
                "id": "coq",
                "name": "Coq",
                "description": "Interactive theorem prover with dependent types",
                "website": "https://coq.inria.fr/"
            }
        ]
    }


@app.get("/api/v1/models")
async def get_supported_models():
    """Get list of supported LLM models."""
    return {
        "models": [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "provider": "OpenAI",
                "description": "Most capable model for complex reasoning"
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "OpenAI",
                "description": "Fast and efficient model"
            },
            {
                "id": "claude-3-opus",
                "name": "Claude 3 Opus",
                "provider": "Anthropic", 
                "description": "Highly capable model for mathematical reasoning"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)