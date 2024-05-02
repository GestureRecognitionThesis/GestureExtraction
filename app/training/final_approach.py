import json
import os

import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.saving.saving_api import save_model, load_model
from keras.src.utils import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score

from .utils import string_to_float32


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
            float_value = string_to_float32(landmark_value)
            landmarks.append(float_value)
        sequences.append(landmarks)
    return sequences


def transform_data_to_sequence_combine(data: dict):
    sequences = []
    for frame_data in data.values():
        landmarks = []
        for landmark_value in frame_data.values():
            landmark_value[0][3] = string_to_float32(landmark_value[0][3]) if landmark_value[0][3] != '0' else 0
            if len(landmarks) == 0:
                landmarks = landmark_value[:4]
            else:
                landmarks.extend(landmark_value[:4])
        flattened_list = [item for sublist in landmarks for item in sublist]
        sequences.append(flattened_list)
    return sequences


def load_coordinate_data(save: bool = False, amount: int = 25, metrics: bool = False):
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
    history = model.fit(padded_sequences, labels, epochs=10, batch_size=1)
    if metrics:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        print(f"Final Loss: {loss[-1]}, Final Accuracy: {accuracy[-1]}")
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predicted_labels, average='weighted')
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted')
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    if save:
        file_name = f"coordinates_{amount}.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


def load_graph_data(save: bool = False, amount: int = 25, metrics: bool = False):
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
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 21)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(padded_sequences, labels, epochs=10, batch_size=1)
    if metrics:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        print(f"Final Loss: {loss[-1]}, Final Accuracy: {accuracy[-1]}")
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predicted_labels, average='weighted')
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted')
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    if save:
        file_name = f"graphs_{amount}.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


def load_combined_data(save: bool = False, amount: int = 25, metrics: bool = False):
    files: list = os.listdir(f'./data/final/combined/{amount}')  # change after testing
    print(f"Amount of files: {len(files)}")
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = f'./data/final/combined/{amount}/{file}'
        data, file_name = load_json_data(data_path)
        if "can" in file_name:
            labels.append(0)
        elif "peace" in file_name:
            labels.append(1)
        elif "thumb" in file_name:
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence_combine(data)
        sequences.append(result)
    labels = np.array(labels)
    max_seq_length = max(len(seq) for seq in sequences)
    print(f"Max sequence length: {max_seq_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 84)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(padded_sequences, labels, epochs=10, batch_size=1)
    if metrics:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        print(f"Final Loss: {loss[-1]}, Final Accuracy: {accuracy[-1]}")
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predicted_labels, average='weighted')
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted')
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    if save:
        file_name = f"combined_{amount}.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


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


def test_model_prediction(prefix: str = '', suffix: str = ''):
    model_path = f"{prefix}_{suffix}.keras"
    model = load_model(model_path)
    random_number = np.random.randint(1, int(suffix))
    print(f"Random number: {random_number}")
    data_path_list = [
        f"data/final/{prefix}/{suffix}/can{random_number}.json",
        f"data/final/{prefix}/{suffix}/peace{random_number}.json",
        f"data/final/{prefix}/{suffix}/thumb{random_number}.json"
    ]
    for path in data_path_list:
        data, file_name = load_json_data(path)
        result = None
        if 'coordinates' in prefix:
            result = transform_data_to_sequence_coordinates(data)
        elif 'graphs' in prefix:
            result = transform_data_to_sequence_graphs(data)
        elif 'combined' in prefix:
            result = transform_data_to_sequence_combine(data)
        if result is None:
            raise ValueError("Invalid prefix")
        sequences = [result]
        max_seq_length = sequence_lengths[f"{prefix}{suffix}"]
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")
        prediction = model.predict(padded_sequences)
        predicted_labels = np.argmax(prediction, axis=1)
        class_labels = {0: "Can", 1: "Peace", 2: "Thumb"}
        print("raw prediction: ", predicted_labels)
        print(f"Predicted label: {class_labels[predicted_labels[0]]}")


sequence_lengths: dict = {
    'coordinates25': 48,
    'graphs25': 47,
    'combined25': 48,
    'coordinates50': 48,
    'graphs50': 47,
    'combined50': 48,
    'coordinates100': 51,
    'graphs100': 50,
    'combined100': 51,
}