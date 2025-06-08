from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, Literal
from core.orchestrator import MCPOrchestrator
from api.models import UnifiedQueryRequest
from api.service_management import get_orchestrator
from fastapi.responses import StreamingResponse
import uuid, json, logging

router = APIRouter()
logger = logging.getLogger(__name__)

class StreamableHTTPResponse(StreamingResponse):
    def __init__(self, query, orchestrator, mode="react", stream_type="step", status_code=200):
        self.query = query
        self.orchestrator = orchestrator
        self.mode = mode
        self.stream_type = stream_type
        media_type = "text/event-stream"
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": media_type
        }
        super().__init__(
            content=self.stream_generator(),
            status_code=status_code,
            media_type=media_type,
            headers=headers
        )
    async def stream_generator(self):
        try:
            if self.stream_type == "step":
                async for response in self.orchestrator.stream_process_query(self.query):
                    event_id = f"event-{uuid.uuid4()}"
                    yield f"id: {event_id}\ndata: {json.dumps(response)}\n\n"
            elif self.stream_type == "token":
                async for response in self.orchestrator.stream_process_query_token(self.query):
                    event_id = f"event-{uuid.uuid4()}"
                    yield f"id: {event_id}\ndata: {json.dumps(response)}\n\n"
            else:
                error_msg = {"error": f"Unsupported stream type: {self.stream_type}"}
                yield f"data: {json.dumps(error_msg)}\n\n"
        except Exception as e:
            logger.error(f"Error in stream generator: {e}", exc_info=True)
            error_response = {"is_final": True, "result": f"Error processing streaming query: {str(e)}"}
            yield f"data: {json.dumps(error_response)}\n\n"

@router.post("/mcp")
async def unified_query_endpoint(
    payload: UnifiedQueryRequest,
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    logger.info(f"Received unified query: '{payload.query[:50]}...', mode: {payload.mode}, stream_type: {payload.stream_type}")
    try:
        if not payload.stream_type:
            result = await orchestrator.process_unified_query(
                query=payload.query,
                mode=payload.mode,
                include_trace=payload.include_trace
            )
            if isinstance(result, str) and result.startswith("Error:"):
                logger.error(f"Error processing query: {result}")
                raise HTTPException(status_code=500, detail=result)
            return {"result": result}
        else:
            return StreamableHTTPResponse(
                query=payload.query,
                mode=payload.mode,
                stream_type=payload.stream_type,
                orchestrator=orchestrator
            )
    except ValueError as ve:
        logger.error(f"Invalid request parameters: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing unified query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 
