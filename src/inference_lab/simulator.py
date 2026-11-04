"""Synthetic SSE fixture, never a source of real inference performance evidence."""
import asyncio
import json
import os
import sys
import threading
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import uvicorn

app = FastAPI()


@app.get('/health')
async def health():
    return {"status": "simulated"}


@app.get('/metrics')
async def metrics():
    return PlainTextResponse('simulated_fixture 1\n')


@app.post('/v1/completions')
@app.post('/v1/chat/completions')
async def completion(request: Request):
    data = await request.json()
    fault = os.getenv("SIM_FAULT", "none")
    if fault == "http503":
        return JSONResponse({"error": "controlled simulated overload"}, status_code=503)
    if fault == "timeout":
        await asyncio.sleep(5)
    is_completion = "prompt" in data
    words = ["Synthetic", " output", ":", " no", " GPU", " evidence", "."]

    if is_completion:
        words = [" token"] * data["max_tokens"]

    async def stream():
        for word in words[:data["max_tokens"]]:
            await asyncio.sleep(0.002 if is_completion else 0.015)
            yield 'data: ' + json.dumps({"choices": [{"text": word} if is_completion else {"delta": {"content": word}}]}) + '\n\n'
        yield 'data: ' + json.dumps({"choices": [], "usage": {
            "prompt_tokens": len(data["prompt"]) if is_completion else len(data["messages"][0]["content"].split()),
            "completion_tokens": min(len(words), data["max_tokens"]),
        }}) + '\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == '__main__':
    timer = threading.Timer(int(sys.argv[2]), lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
