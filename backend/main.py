from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any

from core.config import settings
from core.database import mongodb
from api.v1.api import api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("anya_api.log") if settings.is_production() else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    logger.info("🚀 Starting Anya API...")
    
    try:
        await mongodb.connect()
        logger.info("✅ Application startup completed")
        
        yield  
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("Shutting down Anya API...")
        await mongodb.disconnect()
        logger.info("👋 Application shutdown completed")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.is_development() else None,
    redoc_url="/redoc" if settings.is_development() else None,
    openapi_url="/openapi.json" if settings.is_development() else None,
    lifespan=lifespan
)

# Security middleware
if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["https://anya.ai"]
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Request-ID"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    request_id = f"req-{int(start_time * 1000)}"
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        logger.warning(f"⏱️ Slow request: {request.method} {request.url.path} took {process_time:.2f}s")
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    
    if settings.is_development():
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(exc),
                "type": type(exc).__name__,
                "path": str(request.url.path)
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "message": "An unexpected error occurred"
            }
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

# Include API router
app.include_router(
    api_router, 
    prefix=settings.API_PREFIX
)

@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root endpoint - API information"""
    return {
        "message": "Anya AI News API says welcome",
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs_url": "/docs" if settings.is_development() else None,
        "status": "online",
        "environment": settings.ENVIRONMENT
    }

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        db_health = await mongodb.health_check()
        
        return {
            "status": "healthy" if db_health["status"] == "healthy" else "degraded",
            "service": "anya-api-gateway",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "database": db_health,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f" Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "anya-api-gateway",
            "error": str(e),
            "timestamp": time.time()
        }

if settings.is_development():
    @app.get("/debug/config", tags=["Debug"])
    async def debug_config() -> Dict[str, Any]:
        """Debug endpoint to check configuration (development only)"""
        return {
            "app_name": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "database_name": settings.MONGODB_NAME,
            "api_version": settings.API_VERSION,
            "allowed_origins": settings.ALLOWED_ORIGINS,
            "jwt_algorithm": settings.JWT_ALGORITHM,
            "google_client_configured": bool(settings.GOOGLE_CLIENT_ID)
        }

# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.is_development(),
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
