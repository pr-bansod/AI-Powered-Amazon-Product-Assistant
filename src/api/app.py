import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.api.endpoint import api_router
from api.api.middleware import RequestIDMiddleware
from api.core.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(RequestIDMiddleware)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint that shows welcome message"""
    return {"message": "API"}


