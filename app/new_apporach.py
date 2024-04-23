import json
import os

import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.saving.saving_api import save_model, load_model
from keras.src.utils import pad_sequences


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


def load_data(save: bool = False):
    files: list = os.listdir('./data/test/coordinates/')
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = './data/test/coordinates/' + file
        data, file_name = load_json_data(data_path)
        if "can" in file_name:
            print("appened 0 for file name: " + file_name)
            labels.append(0)
        elif "peace" in file_name:
            print("appened 1 for file name: " + file_name)
            labels.append(1)
        elif "thumb" in file_name:
            print("appened 2 for file name: " + file_name)
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence(data)
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
        file_name = "new_approach_model.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


def custom_predict():
    model = load_model('new_approach_model.keras')
    data_path = "data/test/coordinates/can3.json"
    data, file_name = load_json_data(data_path)
    max_seq_length = 48
    result = transform_data_to_sequence(data)
    sequences = [result]
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_labels = np.argmax(prediction, axis=1)
    print(predicted_labels)


def run(load: bool = False):
    if load:
        load_data(save=False)
    else:
        custom_predict()


if __name__ == '__main__':
    run(load=True)
