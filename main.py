# main.py
from fastapi import FastAPI
from routes.predict import router as predict_router
from src.config import settings

# Validate environment variables on startup
settings.validate()

app = FastAPI(
    title="Hotel Financial Risk Predictive API",
    description="API for predicting hotel financial risk using GNN and ensemble models",
    version="1.0.0"
)

app.include_router(predict_router)

@app.get("/")
def root():
    return {
        "message": "Hotel Financial Risk Predictive API is running.",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )