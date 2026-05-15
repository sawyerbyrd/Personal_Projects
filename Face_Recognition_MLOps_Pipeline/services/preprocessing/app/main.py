"""Preprocessing FastAPI service entrypoint."""

from fastapi import FastAPI

from app.api import health, preprocess

app = FastAPI(title="Preprocessing", version="1.0.0")
app.include_router(health.router)
app.include_router(preprocess.router)
