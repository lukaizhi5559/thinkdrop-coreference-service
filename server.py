"""
Coreference Resolution MCP Service
Resolves pronouns and references in conversation messages
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.middleware.validation import validate_mcp_request, validate_api_key
from src.routes.resolve import router as resolve_router

# Load environment variables from service directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Coreference Resolution Service",
    description="MCP service for resolving pronouns and references in conversations",
    version="1.0.0"
)

# CORS middleware
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "coreference",
        "version": "1.0.0"
    }

# Include routers
app.include_router(resolve_router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "version": "mcp.v1",
            "status": "error",
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc)
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', 3005))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"🚀 Starting Coreference Service on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )
