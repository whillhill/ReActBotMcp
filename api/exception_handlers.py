from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_messages = []
    for error in errors:
        loc = " -> ".join([str(l) for l in error["loc"] if l != "body"])
        msg = error["msg"]
        error_messages.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Validation error", "errors": error_messages},
    ) 
