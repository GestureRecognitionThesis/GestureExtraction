import tempfile
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model
from training import extract, process_mp, fit_data_to_sequence, prepare_sequences
import numpy as np

model_router = APIRouter(prefix="/model")


def load_and_use_model():
    print("Loading model")
    return load_model('gesture_recognition_model.keras')


model = load_and_use_model()


@model_router.get("/")
async def get_model():
    return {"model": model.summary()}


@model_router.post("/uploadVideo")
async def upload_video(video_file: UploadFile = File(...)):
    try:
        # Read the binary data from the uploaded file
        video_data = await video_file.read()

        # Write the binary data to a .mov file
        with open("uploaded_video.mov", "wb") as f:
            f.write(video_data)

        return JSONResponse(content={"status": "File uploaded successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@model_router.get("/predict")
async def predict(video_file: UploadFile = UploadFile(...)):
    # Read the binary data from the uploaded file
    video_data = await video_file.read()
    temp_video_file_path = create_temp_file_return_path(video_data)
    # Extract frames from the video
    frames = extract(temp_video_file_path)
    all_data: list = []
    for frame in frames:
        all_data.append(process_mp(frame))
    # Convert the data to a dictionary
    data_dict = convert_list_data_to_dict(all_data)
    result = fit_data_to_sequence(data_dict)
    sequences: list = [result]
    padded_sequences, labels, max_length = prepare_sequences(sequences, [])
    padded_sequences = np.array(padded_sequences)
    prediction = model.predict(padded_sequences)
    predicted_labels = np.argmax(prediction, axis=1)
    class_labels = {0: "Can", 1: "Peace", 2: "Thumb"}  # Update this dictionary with your class labels

    # Map predicted class indices to their corresponding labels
    predicted_labels = [class_labels[idx] for idx in predicted_labels]
    # return the most common prediction
    return JSONResponse(content={"prediction": max(set(predicted_labels), key=predicted_labels.count)}, status_code=200)


def create_temp_file_return_path(video_data):
    temp_video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_file.write(video_data)
    temp_video_file.close()

    return temp_video_file.name


def convert_list_data_to_dict(data: list):
    dict_data = {}
    for i, item in enumerate(data, start=1):
        frame_data_by_landmark = {}
        for frame_data in item:
            landmark = "Landmark" + str(frame_data.landmark + 1)
            if landmark not in frame_data_by_landmark:
                frame_data_by_landmark[landmark] = []

            frame_data_list = [
                frame_data.relative[0],
                frame_data.relative[1],
                frame_data.relative[2],
            ]
            frame_data_by_landmark[landmark].append(frame_data_list)
        dict_data[f"frame{i}"] = frame_data_by_landmark
    return dict_data
