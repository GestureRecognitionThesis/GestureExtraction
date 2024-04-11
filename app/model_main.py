import numpy as np

from training import *
from keras.models import load_model
import json
import os
from tqdm import tqdm


# all_data[0] is the first frame
# all_data[0][0] is the first landmark
# the length of all_data[0] is equal to the total amount of landmarks processed in the frame

def calculate_graphs(data: list):
    for i in range(len(data)):  # every frame
        for j in range(len(data[i])):  # every landmark
            if i == 0:
                data[i][j].direct_graph = "firstFrame"
                continue
            data[i][j].direct_graph = calculate_line_equation(data[i - 1][j], data[i][j])


def save_to_json(path: str, data: list, file_name: str, v2: bool = False):
    # Initialize an empty dictionary to store the JSON data
    json_data = {}

    # Iterate over each item in the data list
    for i, item in enumerate(data, start=1):
        # Initialize a dictionary to store frame data for each landmark
        frame_data_by_landmark = {}

        # Iterate over each FrameData object in the item
        for frame_data in item:
            if frame_data.direct_graph == "firstFrame" and v2:
                continue
            landmark = "Landmark" + str(frame_data.landmark + 1)
            if v2:
                frame_data_by_landmark[landmark] = frame_data.direct_graph
            else:
                if landmark not in frame_data_by_landmark:
                    frame_data_by_landmark[landmark] = []

                frame_data_list = [
                    frame_data.relative[0],
                    frame_data.relative[1],
                    frame_data.relative[2],
                ]
                frame_data_by_landmark[landmark].append(frame_data_list)

        # Add frame data by landmark to the JSON data dictionary
        if v2:
            if i == 1:
                continue
        json_data[f"frame{i}"] = frame_data_by_landmark

    # Write the dictionary to a JSON file
    with open(path + file_name + ".json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def extract_and_save_data(subpath: str = '', v2: bool = False, test: bool = False):
    if test:
        files: list = find_test_video_file_names()
    else:
        files: list = find_video_file_names()
    progres_bar = tqdm(total=len(files) * 3, unit='iteration')  # because we have 3 steps
    for file in files:
        if str(file).split(".")[1] != "MOV":
            continue
        # print("Processing: " + file)
        progres_bar.set_description(f"Processing file {file}")
        if test:
            video_path = f'./data/videos/test/{file}'
        else:
            video_path = f'./data/videos/valid/{file}'
        data_path = f'./data/train/{subpath}/'
        progres_bar.update(1)
        progres_bar.set_description(f"Extracting frames from {file}")
        extracted_frames = extract(video_path)
        all_data: list = []
        progres_bar.update(1)
        progres_bar.set_description(f"MediaPipe processing {file}")
        for frame in extracted_frames:
            all_data.append(process_mp(frame))
        progres_bar.update(1)
        if v2:
            progres_bar.set_description(f"Calculating graphs for file {file}")
            calculate_graphs(all_data)
        progres_bar.set_description(f"Saving to {file} to json")
        save_to_json(data_path, all_data, str(file).split(".")[0], v2)
    progres_bar.close()
    """
    video_path = '../data/videos/can1.MOV'
    data_path = './data/'
    extracted_frames = extract(video_path)
    all_data: list = []
    for frame in extracted_frames:
        all_data.append(process_mp(frame))

    print(all_data)
    calculate_graphs(all_data)

    save_to_json(data_path, all_data)
    """


# Load a single video, without saving to a json file, and make the model predict the gesture
def load_single_video_and_predict():
    video_path = '../data/videos/can1.MOV'
    extracted_frames = extract(video_path)
    all_data: list = []
    for frame in extracted_frames:
        all_data.append(process_mp(frame))
    data_dict = convert_list_data_to_dict(all_data)
    print(data_dict)


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


def load_data_input_to_model(save: bool, sub_path: str = ''):
    # data path to json file
    files: list = find_json_file_names(sub_path)
    sequences: list = []
    labels: list = []
    for file in files:
        print("Processing: " + file)
        data_path = './data/' + sub_path + '/' + file
        data: dict
        file_name: str
        data, file_name = load_json(data_path)
        labels.append(file_name)
        result = fit_data_to_sequence(data)
        sequences.append(result)
    define_and_train_model(sequences, labels, save)


def find_json_file_names(sub_path: str = ''):
    data_path = './data/' + sub_path + '/'
    return os.listdir(data_path)


def find_video_file_names():
    data_path = './data/videos/valid/'
    videos = os.listdir(data_path)
    return videos


def find_test_video_file_names():
    data_path = './data/videos/test/'
    return os.listdir(data_path)


def load_and_use_model():
    # Load the model
    model = load_model('gesture_recognition_model.keras')
    # Predict
    data_path = './data/test/can1.json'
    data: dict
    file_name: str
    data, file_name = load_json(data_path)
    print(data)
    sequences: list = []
    labels: list = [file_name]
    result = fit_data_to_sequence(data)
    sequences.append(result)
    padded_sequences, labels, max_length = prepare_sequences(sequences, labels)
    padded_sequences = np.array(padded_sequences)
    prediction = model.predict(padded_sequences)
    predicted_labels = np.argmax(prediction, axis=1)
    class_labels = {0: "Can", 1: "Peace", 2: "Thumb"}  # Update this dictionary with your class labels

    # Map predicted class indices to their corresponding labels
    predicted_labels = [class_labels[idx] for idx in predicted_labels]
    print("Predicted Labels:", predicted_labels)
    print("Raw Prediction:", prediction)


# list [ [ 21 landmarks in here (FrameData) ], [ 21 landmarks in here (FrameData) ], [ 21 landmarks in here (FrameData) ] ]
if __name__ == '__main__':
    # extract_and_save_data(subpath="coordinates")
    extract_and_save_data(subpath="graphs", v2=True)
    # load_data_input_to_model(False, 'train/coordinates')
    # load_and_use_model()
    # load_single_video_and_predict()
