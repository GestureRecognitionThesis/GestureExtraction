from fastapi import APIRouter

status_router = APIRouter(prefix="/status")


@status_router.get("/")
async def get_status():
    return {"status": "ok"}
