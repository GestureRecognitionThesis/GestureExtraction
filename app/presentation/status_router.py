from fastapi import APIRouter

status_router = APIRouter(prefix="/status")

some_value = 1


@status_router.get("/")
async def get_status():
    return {"status": some_value}
