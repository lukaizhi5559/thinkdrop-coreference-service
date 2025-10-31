"""
Coreference Resolution Routes
"""

import logging
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from ..middleware.validation import validate_api_key, validate_mcp_request
from ..coreference_resolver import CoreferenceResolver

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize resolver (lazy loading)
resolver = None

def get_resolver():
    global resolver
    if resolver is None:
        resolver = CoreferenceResolver()
    return resolver

# Request/Response models
class ConversationMessage(BaseModel):
    role: str
    content: str

class ResolvePayload(BaseModel):
    message: str
    conversationHistory: List[ConversationMessage] = Field(default_factory=list)
    options: Optional[Dict] = Field(default_factory=dict)

class MCPRequest(BaseModel):
    version: str
    service: str
    action: str
    requestId: str
    payload: ResolvePayload

class Replacement(BaseModel):
    original: str
    resolved: str
    confidence: float
    position: int

class ResolveResponse(BaseModel):
    originalMessage: str
    resolvedMessage: str
    replacements: List[Replacement]
    method: str

@router.post("/resolve")
async def resolve_coreferences(
    request: Request,
    mcp_request: MCPRequest,
    _: bool = Depends(validate_api_key)
):
    """
    Resolve coreferences in a message using conversation history
    """
    try:
        # Validate MCP request
        validate_mcp_request(mcp_request.dict())
        
        logger.info(f"📥 Resolving coreferences for: {mcp_request.payload.message[:50]}...")
        
        # Get resolver
        resolver = get_resolver()
        
        # Convert conversation history to simple format
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in mcp_request.payload.conversationHistory
        ]
        
        # Resolve coreferences
        result = resolver.resolve(
            message=mcp_request.payload.message,
            conversation_history=history,
            options=mcp_request.payload.options
        )
        
        logger.info(f"✅ Resolved: {result['resolvedMessage'][:50]}...")
        
        # Return MCP response
        return {
            "version": "mcp.v1",
            "service": "coreference",
            "action": "resolve",
            "requestId": mcp_request.requestId,
            "status": "ok",
            "data": result,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"❌ Resolution failed: {e}", exc_info=True)
        return {
            "version": "mcp.v1",
            "service": "coreference",
            "action": "resolve",
            "requestId": mcp_request.requestId,
            "status": "error",
            "data": None,
            "error": {
                "code": "RESOLUTION_FAILED",
                "message": str(e)
            }
        }
