"""Inference FastAPI service entrypoint."""

from fastapi import FastAPI

from app.api import health, models, predict

app = FastAPI(title="Inference", version="1.0.0")
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
