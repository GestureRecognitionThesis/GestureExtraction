from fastapi import FastAPI
import uvicorn

from presentation import status_router, model_router

fastApi = FastAPI()
fastApi.include_router(status_router)
fastApi.include_router(model_router)

if __name__ == "__main__":
    uvicorn.run(fastApi, host="0.0.0.0", port=8000)

