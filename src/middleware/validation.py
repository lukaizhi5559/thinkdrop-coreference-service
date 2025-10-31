"""
MCP Request Validation Middleware
"""

import os
from fastapi import Request, HTTPException
from typing import Dict, Any

def validate_api_key(request: Request) -> bool:
    """Validate API key from request headers"""
    # Accept both X-API-Key and Authorization headers (MCP client uses Authorization)
    api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
    expected_key = os.getenv('API_KEY')
    
    if not expected_key:
        # If no API key is configured, allow all requests (development mode)
        return True
    
    # Strip "Bearer " prefix if present
    if api_key and api_key.startswith('Bearer '):
        api_key = api_key[7:]
    
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    
    return True

def validate_mcp_request(data: Dict[str, Any]) -> bool:
    """Validate MCP request format"""
    required_fields = ['version', 'service', 'action', 'requestId', 'payload']
    
    for field in required_fields:
        if field not in data:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    if data['version'] != 'mcp.v1':
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported MCP version: {data['version']}"
        )
    
    if data['service'] != 'coreference':
        raise HTTPException(
            status_code=400,
            detail=f"Invalid service: {data['service']}"
        )
    
    return True
