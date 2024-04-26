import json
import os
import re

import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.saving.saving_api import save_model, load_model
from keras.src.utils import pad_sequences
from utils import string_to_float32


def load_json_data(path: str) -> tuple[dict, str]:
    with open(path, "r") as json_file:
        data = json.load(json_file)
    # file_name without .json
    file_name = path.split("/")[-1].split(".")[0]
    # remove all numbers from the text
    file_name = ''.join([i for i in file_name if not i.isdigit()])
    return data, file_name


def transform_data_to_sequence_coordinates(data: dict):
    sequences: list = []
    for frame in data.values():
        landmarks = []
        for landmark_data in frame.values():
            if len(landmarks) == 0:
                landmarks = landmark_data[:3]
            else:
                landmarks.extend(landmark_data[:3])
        flattened_list = [item for sublist in landmarks for item in sublist]
        sequences.append(flattened_list)
    return sequences


def transform_data_to_sequence_graphs(data: dict):
    sequences = []
    for frame_data in data.values():
        landmarks = []
        for landmark_value in frame_data.values():
            float_value = extract_coefficients(landmark_value)
            landmarks.append(float_value)
        sequences.append(landmarks)
    return sequences


def extract_coefficients(expression: str) -> float:
    # Split the expression by 'x' and '+' to extract coefficients
    parts = expression.split('x')

    # Extract and process 'a' coefficient
    a_str = parts[0].strip().replace(".", "").replace("-", "") if parts[
        0].strip() else "0"  # coefficient of x without decimal and minus

    # Extract and process 'b' constant term
    b_str = "0"
    if len(parts) > 1:
        constant_part = parts[1].strip()
        if '+' in constant_part:
            b_str = constant_part.split('+')[1].strip().replace("-", "")
        elif "-" in constant_part:
            b_str = "-" + constant_part.split('-')[1].strip()

    combined_str = a_str + b_str  # Concatenate 'a' and 'b' strings
    return float(combined_str)  # Convert concatenated string to float


def load_coordinate_data(save: bool = False, amount: int = 25):
    files: list = os.listdir(f'./data/final/coordinates/{amount}/')
    print(f"Amount of files: {len(files)}")
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = f'./data/final/coordinates/{amount}/{file}'
        data, file_name = load_json_data(data_path)
        if "can" in file_name:
            labels.append(0)
        elif "peace" in file_name:
            labels.append(1)
        elif "thumb" in file_name:
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence_coordinates(data)
        sequences.append(result)
    labels = np.array(labels)
    max_seq_length = max(len(seq) for seq in sequences)
    print(f"Max sequence length: {max_seq_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 63)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, labels, epochs=10, batch_size=1)

    if save:
        file_name = f"coordinates_{amount}.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


def load_graph_data(save: bool = False, amount: int = 25):
    files: list = os.listdir(f'./data/final/graphs/{amount}/')
    print(f"Amount of files: {len(files)}")
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = f'./data/final/graphs/{amount}/{file}'
        data, file_name = load_json_data(data_path)
        if "can" in file_name:
            labels.append(0)
        elif "peace" in file_name:
            labels.append(1)
        elif "thumb" in file_name:
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence_graphs(data)
        sequences.append(result)
    labels = np.array(labels)
    max_seq_length = max(len(seq) for seq in sequences)
    print(f"Max sequence length: {max_seq_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')


def custom_predict_coordinates():
    model = load_model('new_approach_model.keras')
    data_path = "data/test/coordinates/can3.json"
    data, file_name = load_json_data(data_path)
    max_seq_length = 48
    result = transform_data_to_sequence_coordinates(data)
    sequences = [result]
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_labels = np.argmax(prediction, axis=1)
    print(predicted_labels)


def run(arg: str = ''):
    if arg == '':
        raise ValueError("Invalid argument")

    amount = int(arg.split()[0])
    saving = True if 'save' in arg else False
    if 'load' in arg:
        if 'coordinates' in arg:
            load_coordinate_data(save=saving, amount=amount)
        elif 'graphs' in arg:
            load_graph_data(save=saving, amount=amount)
    else:
        custom_predict_coordinates()


if __name__ == '__main__':
    run(arg="25 load graphs")
