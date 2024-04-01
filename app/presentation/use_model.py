from fastapi import APIRouter
from keras.models import load_model

model_router = APIRouter(prefix="/model")

model = load_model('gesture_recognition_model.keras')


@model_router.get("/")
async def get_model():
    return {"model": model.summary()}
