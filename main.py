# main.py
from fastapi import FastAPI
from routes.predict import router as predict_router


app = FastAPI(title="Hotel Financial Risk Predictive API")
app.include_router(predict_router)

@app.get("/")
def root():
    return {"message": "Hotel Financial Risk Predictive API is running."}