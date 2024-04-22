import json
import os

import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense


def load_json_data(path: str) -> tuple[dict, str]:
    with open(path, "r") as json_file:
        data = json.load(json_file)
    # file_name without .json
    file_name = path.split("/")[-1].split(".")[0]
    print(f"Loaded data from {file_name}")
    # remove all numbers from the text
    file_name = ''.join([i for i in file_name if not i.isdigit()])
    return data, file_name

def transform_data_to_sequence(data: dict):
    sequences: list = []
    i = 0
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

def load_data():
    files: list = os.listdir('./data/test/coordinates/')
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = './data/test/coordinates/' + file
        data, file_name = load_json_data(data_path)
        class_labels = {"can": 0, "peace": 1, "thumb": 2}
        labels.append(0)
        result = transform_data_to_sequence(data)
        sequences.append(result)
    labels = np.array(labels)
    max_seq_length = max(len(seq) for seq in sequences)
    padded_sequences = np.array(
        [np.pad(seq, ((0, max_seq_length - len(seq)), (0, 0)), mode='constant') for seq in sequences])

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 63)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, labels, epochs=10, batch_size=1)

if __name__ == '__main__':
    load_data()
