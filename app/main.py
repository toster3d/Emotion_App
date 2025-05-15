import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import time

from app.core.settings import settings
from app.api import api_router
from app.core import lifespan_model_loading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app with lifespan for model loading
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan_model_loading,
)

# Add middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve static files from frontend build directory if it exists
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.get("/")
async def root():
    """Root endpoint that redirects to the API documentation."""
    return {"message": "Welcome to Audio Emotion Detection API", "docs_url": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 

# import logging
# import uvicorn
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware

# from app.api import endpoints
# from app.core import settings

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     handlers=[
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)

# # Create FastAPI application
# app = FastAPI(
#     title=settings.API_TITLE,
#     description=settings.API_DESCRIPTION,
#     version=settings.API_VERSION,
#     openapi_url=settings.OPENAPI_URL,
#     docs_url=settings.DOCS_URL,
# )

# # Add CORS middleware for frontend integration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include API router
# app.include_router(endpoints.router, prefix="/api/v1")

# # Global exception handler
# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={"detail": f"Internal server error: {str(exc)}"}
#     )

# @app.get("/")
# async def root():
#     return {
#         "message": "Welcome to the Emotion Recognition API",
#         "docs": "/docs",
#         "health": "/api/v1/health"
#     }

# if __name__ == "__main__":
#     logger.info(f"Starting Emotion Recognition API on {settings.ENVIRONMENT} environment")
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)