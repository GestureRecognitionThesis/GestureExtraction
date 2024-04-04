from fastapi import FastAPI
import uvicorn
from starlette.middleware.trustedhost import TrustedHostMiddleware

from presentation import status_router, model_router

fastApi = FastAPI()
fastApi.include_router(status_router)
fastApi.include_router(model_router)

def runApi():
    uvicorn.run(fastApi, host="0.0.0.0", port=8000, ws_max_size=6553500, log_level="info")


if __name__ == "__main__":
    runApi()

