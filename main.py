# main.py
from fastapi import FastAPI
from routes.predict import router as predict_router
from routes.new_data import router as new_data_router
from routes.company_data import router as company_data_router
from routes.knowledge_graph import router as knowledge_graph_router
# from src.config import settings
from fastapi.middleware.cors import CORSMiddleware


# # Validate environment variables on startup
# settings.validate()

app = FastAPI(
    title="Hotel Financial Risk Predictive API",
    description="API for predicting hotel financial risk using GNN and ensemble models",
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