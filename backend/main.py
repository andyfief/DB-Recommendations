from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.recommend import router

app = FastAPI(title="DrinkAdvisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup() -> None:
    from backend.lib.db import ensure_initialized
    ensure_initialized()
