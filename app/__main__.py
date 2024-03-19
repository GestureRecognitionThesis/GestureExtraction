import numpy as np

from extraction import extract
from processing import process_mp, calculate_line_equation
from prediction import *
from keras.models import load_model
import json


# all_data[0] is the first frame
# all_data[0][0] is the first landmark
# the length of all_data[0] is equal to the total amount of landmarks processed in the frame

def calculate_graphs(data: list):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if i == 0:
                data[i][j].direct_graph = "firstFrame"
                continue
            data[i][j].direct_graph = calculate_line_equation(data[i - 1][3], data[i][3], i)


def save_to_json(path: str, data: list):
    # Initialize an empty dictionary to store the JSON data
    json_data = {}

    # Iterate over each item in the data list
    for i, item in enumerate(data, start=1):
        # Initialize a dictionary to store frame data for each landmark
        frame_data_by_landmark = {}

        # Iterate over each FrameData object in the item
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

        # Add frame data by landmark to the JSON data dictionary
        json_data[f"frame{i}"] = frame_data_by_landmark

    # Write the dictionary to a JSON file
    with open(path + "can1.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def extract_and_save_data():
    video_path = '../data/videos/can1.MOV'
    data_path = './data/'
    extracted_frames = extract(video_path)
    all_data: list = []
    for frame in extracted_frames:
        all_data.append(process_mp(frame))

    calculate_graphs(all_data)

    save_to_json(data_path, all_data)


def load_data_input_to_model(save: bool):
    # data path to json file
    data_path = './data/can1.json'
    data: dict
    file_name: str
    data, file_name = load_json(data_path)
    sequences: list = []
    labels: list = [file_name, file_name]
    result = fit_data_to_sequence(data)
    sequences.append(result)
    sequences.append(result)
    define_and_train_model(sequences, labels, save)


def load_and_use_model():
    # Load the model
    model = load_model('gesture_recognition_model.keras')
    # Predict
    data_path = './data/can1.json'
    data: dict
    file_name: str
    data, file_name = load_json(data_path)
    sequences: list = []
    labels: list = [file_name, file_name]
    result = fit_data_to_sequence(data)
    sequences.append(result)
    padded_sequences, labels, max_length = prepare_sequences(sequences, labels)
    padded_sequences = np.array(padded_sequences)
    prediction = model.predict(padded_sequences)
    predicted_labels = np.argmax(prediction, axis=1)
    class_labels = {0: "Can"}  # Update this dictionary with your class labels

    # Map predicted class indices to their corresponding labels
    predicted_labels = [class_labels[idx] for idx in predicted_labels]
    print("Predicted Labels:", predicted_labels)


# list [ [ 21 landmarks in here (FrameData) ], [ 21 landmarks in here (FrameData) ], [ 21 landmarks in here (FrameData) ] ]
if __name__ == '__main__':
    # extract_and_save_data()
    #load_data_input_to_model(False)
    load_and_use_model()
    # print("length of data: " + str(len(all_data)))
