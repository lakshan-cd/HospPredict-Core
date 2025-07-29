# main.py
from fastapi import FastAPI
from routes.predict import router as predict_router
from routes.new_data import router as new_data_router
from routes.company_data import router as company_data_router
from routes.knowledge_graph import router as knowledge_graph_router
from routes.macro_economic import router as macro_economic_router

# Import news prediction routes
from routes.news.prediction import router as news_prediction_router
from routes.news.analytics import router as news_analytics_router
from routes.news.admin import router as news_admin_router
from routes.news.websocket import router as news_websocket_router

# from src.config import settings
from fastapi.middleware.cors import CORSMiddleware


# # Validate environment variables on startup
# settings.validate()

app = FastAPI(
    title="Hotel Financial Risk Predictive API",
    description="API for predicting hotel financial risk using GNN and ensemble models, including news-based stock price prediction",
    version="1.0.0"
)
app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Or specify ["http://localhost:3000"] for more security
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
  

app.include_router(predict_router, prefix="/api/v1", tags=["predictions"])
app.include_router(new_data_router, prefix="/api/v1", tags=["new-data"])
app.include_router(company_data_router, prefix="/api/v1", tags=["company-data"])
app.include_router(knowledge_graph_router, prefix="/api/v1", tags=["knowledge-graph"])
app.include_router(macro_economic_router, prefix="/api/v1", tags=["macro-economic"])

# Include news prediction routes
app.include_router(news_prediction_router, prefix="/api/v1", tags=["news-prediction"])
app.include_router(news_analytics_router, prefix="/api/v1", tags=["news-analytics"])
app.include_router(news_admin_router, prefix="/api/v1", tags=["news-admin"])
app.include_router(news_websocket_router, prefix="/api/v1", tags=["news-websocket"])

@app.get("/")
def root():
    return {
        "message": "Hotel Financial Risk Predictive API is running.",
        "version": "1.0.0",
        "status": "healthy",
        "features": [
            "GNN-based financial risk prediction",
            "News-based stock price prediction",
            "Real-time WebSocket updates",
            "Analytics and insights",
            "Admin tools and model management"
        ],
        "endpoints": {
            "predictions": "/api/v1/predictions",
            "new-data": "/api/v1/new-data",
            "company-data": "/api/v1/company-data",
            "knowledge-graph": "/api/v1/knowledge-graph",
            "macro-economic": "/api/v1/macro-economic",
            "news-prediction": "/api/v1/news",
            "news-analytics": "/api/v1/news/analytics",
            "news-admin": "/api/v1/news/admin",
            "news-websocket": "/api/v1/news/ws"
        }
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