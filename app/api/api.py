from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routers.detect import router
from app.api.utils.load_model import initialize_model
import traceback

# Uygulama çalışma süresi boyunca modelin yüklenmesini sağlar


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_model()

    except Exception as e:
        print(f"Failed to initialize model during startup: {e}")
        traceback.print_exc()
        raise
    yield


# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Chess Pawn Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Router'ları include et
app.include_router(router, prefix="/api", tags=["detection"])

# Root endpoint


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "🏁 Chess Pawn Detection API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/api/health",
        "endpoints": {
            "detect": "/api/detect",
            "detect_visual": "/api/detect-visual",
            "batch_detect": "/api/batch-detect",
            "health": "/api/health",
            "model_info": "/api/model-info"
        },
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "model_type": "PyTorch FasterRCNN ResNet50 FPN"
    }
