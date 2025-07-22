# main.py
from fastapi import FastAPI
from routes.predict import router as predict_router
from routes.new_data import router as new_data_router
# from src.config import settings

# # Validate environment variables on startup
# settings.validate()

app = FastAPI(
    title="Hotel Financial Risk Predictive API",
    description="API for predicting hotel financial risk using GNN and ensemble models",
    version="1.0.0"
)

app.include_router(predict_router, prefix="/api/v1", tags=["predictions"])
app.include_router(new_data_router, prefix="/api/v1", tags=["new-data"])

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
        host="0.0.0.0",
        port=8000,
        reload=True
    )