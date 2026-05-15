"""Retraining FastAPI service entrypoint."""

from fastapi import FastAPI

from app.api import health, retrain

app = FastAPI(title="Retraining", version="1.0.0")
app.include_router(health.router)
app.include_router(retrain.router)
