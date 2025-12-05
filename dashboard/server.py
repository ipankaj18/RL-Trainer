import os
import json
import argparse
import asyncio
import hashlib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="RL Trader Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1_METRICS_TEMPLATE = "models/phase1_jax/training_metrics_{market}.json"
PHASE2_METRICS_TEMPLATE = "results/{market}_jax_phase2_realtime_metrics.json"

# Global state
current_market = "NQ"


def load_metrics(market: str):
    """Load metrics JSON for a market, preferring Phase 2 then Phase 1."""
    p2_path = BASE_DIR / PHASE2_METRICS_TEMPLATE.format(market=market)
    p1_path = BASE_DIR / PHASE1_METRICS_TEMPLATE.format(market=market)

    metrics_file = p2_path if p2_path.exists() else p1_path if p1_path.exists() else None

    if not metrics_file or not metrics_file.exists():
        return None, None

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Attach last_updated from file mtime if missing
    try:
        mtime = metrics_file.stat().st_mtime
    except OSError:
        mtime = None
    if 'last_updated' not in data and mtime:
        data['last_updated'] = mtime

    return data, mtime

@app.get("/api/metrics")
async def get_metrics(market: str = None):
    global current_market
    target_market = market or current_market
    data, _ = load_metrics(target_market)

    if not data:
        return {
            "error": "Metrics file not found",
            "searched_paths": [
                str(BASE_DIR / PHASE1_METRICS_TEMPLATE.format(market=target_market)),
                str(BASE_DIR / PHASE2_METRICS_TEMPLATE.format(market=target_market))
            ],
            "status": "waiting_for_training"
        }

    return data


@app.get("/api/stream")
async def stream_metrics(market: str = None):
    """Simple Server-Sent Events stream for real-time dashboard updates."""
    target_market = market or current_market

    async def event_generator():
        last_hash = None
        while True:
            data, mtime = load_metrics(target_market)
            if data:
                payload = json.dumps(data)
                payload_hash = hashlib.md5(payload.encode()).hexdigest()
                if payload_hash != last_hash:
                    last_hash = payload_hash
                    yield f"data: {payload}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Serve React App
# We expect the build output to be in 'dist' folder relative to this script
DIST_DIR = Path(__file__).parent / "dist"

if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
else:
    print(f"WARNING: React build directory not found at {DIST_DIR}. Only API will work.")
    @app.get("/")
    async def root():
        return {"message": "Dashboard API is running. React frontend not built yet."}

def start_server(host="0.0.0.0", port=8000, market="NQ"):
    global current_market
    current_market = market
    print(f"🚀 Starting Dashboard Server on http://{host}:{port}")
    print(f"📊 Monitoring Market: {market}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--market", default="NQ", help="Market symbol to monitor")
    args = parser.parse_args()
    
    start_server(args.host, args.port, args.market)
