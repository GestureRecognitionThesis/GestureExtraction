import json
from typing import Tuple, Any

import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.utils import pad_sequences
from keras.models import save_model


def load_json(path: str) -> tuple[dict, str]:
    with open(path, "r") as json_file:
        data = json.load(json_file)
    # file_name without .json
    file_name = path.split("/")[-1].split(".")[0]
    print(f"Loaded data from {file_name}")
    # remove all numbers from the text
    file_name = ''.join([i for i in file_name if not i.isdigit()])
    return data, file_name


def fit_data_to_sequence(data: dict):
    sequences: list = []
    i = 0
    for frame in data.values():
        landmarks = []
        for landmark_data in frame.values():
            landmarks.append(list(landmark_data)[:3])  # Extract only the first 3 values
        flattened_list = [item for sublist in landmarks for item in sublist]
        sequences.append(flattened_list)
    return sequences


def prepare_sequences(all_sequences: list, all_sequence_labels: list) -> Tuple[list, list, int]:
    # Combine sequences and labels
    sequences = []
    labels = []

    for i, sequence in enumerate(all_sequences):
        sequences.extend(sequence)
        if str.lower(all_sequence_labels[i]) == "can":
            labels.extend([[1, 0, 0]] * len(sequence))
        elif str.lower(all_sequence_labels[i]) == "peace":
            labels.extend([[0, 1, 0]] * len(sequence))
        elif str.lower(all_sequence_labels[i]) == "thumb":
            labels.extend([[0, 0, 1]] * len(sequence))

    # Pad sequences to ensure equal length
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', dtype="float32")
    print("** MAX LENGTH: ", max_length, " **")

    return padded_sequences, labels, max_length


def prepare_sequences_without_labels(all_sequences: list) -> Tuple[list, int]:
    # Combine sequences and labels
    sequences = []

    for i, sequence in enumerate(all_sequences):
        sequences.extend(sequence)

    # Pad sequences to ensure equal length
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', dtype="float32")
    print("** MAX LENGTH: ", max_length, " **")

    return padded_sequences, max_length


def define_and_train_model(all_sequences: list, all_sequence_labels: list, save: bool = False):
    # Prepare sequences and labels
    padded_sequences, labels, max_length = prepare_sequences(all_sequences, all_sequence_labels)

    # Convert to numpy arrays
    padded_sequences = np.array(padded_sequences)
    labels = np.array(labels)

    # Define the model
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(max_length, 3)),  # Assuming each data point has 4 features
        Dense(3, activation='softmax')  # Assuming 3 output classes: "Fish", "Tank", "Nose"
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, labels, epochs=10, batch_size=1, validation_split=0.2)

    if save:
        print("Model training complete.")
        save_model(model, "gesture_recognition_model.keras")
