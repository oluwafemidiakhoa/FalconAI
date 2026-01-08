"""
REST API for FalconAI Inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from ..config import normalize_config, build_falcon, FalconSpec
from ..core import FalconAI

class InferenceRequest(BaseModel):
    data: Any
    context: Optional[Dict[str, Any]] = None

class InferenceResponse(BaseModel):
    action: Optional[str]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class LearnRequest(BaseModel):
    action: str
    outcome: str
    reward: float
    metadata: Optional[Dict[str, Any]] = None

def create_inference_app(config: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="FALCON-AI Inference API", version="1.0.0")

    # Initialize FalconAI
    app_cfg = normalize_config(config)
    falcon: FalconAI = build_falcon(app_cfg.falcon)

    @app.get("/health")
    async def health():
        status = falcon.get_status()
        return {"status": "ok", "falcon_status": status}

    @app.post("/v1/infer", response_model=InferenceResponse)
    async def infer(request: InferenceRequest):
        try:
            decision = falcon.process(request.data, request.context)
            
            if decision is None:
                return {
                    "action": None,
                    "confidence": 0.0,
                    "reasoning": "No salient event detected",
                    "metadata": {}
                }
            
            return {
                "action": decision.action.value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "metadata": decision.metadata or {}
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/learn")
    async def learn(request: LearnRequest):
        # Note: In a real scenario, we'd need to reconstruct the Decision object 
        # or have a more direct way to feed outcomes to the correction loop.
        # For now, we'll implement a basic observation if possible, 
        # but the current FalconAI.observe requires a full Decision object.
        # This is a placeholder for future extension where we handle ID-based learning.
        
        return {"status": "not_implemented_fully", "message": "Learning endpoint requires state tracking"}

    return app
