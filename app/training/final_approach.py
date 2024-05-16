import csv
import json
import os

import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.saving.saving_api import save_model, load_model
from keras.src.utils import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
from time import perf_counter

from training import process_mp

from . import extract
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


def load_coordinate_data(save: bool = False, amount: int = 25, metrics: bool = False, callback: list = None):
    env_save = True if os.getenv("model_metrics") == "True" else False
    env_time = True if os.getenv("time_metrics") == "True" else False
    print(env_save, env_time)
    time = None
    if env_time:
        time = perf_counter()
    files: list = os.listdir(f'./data/final/coordinates/{amount}/')
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "os.listdir", perf_counter() - time])
        time = perf_counter()
    print(f"Amount of files: {len(files)}")
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = f'./data/final/coordinates/{amount}/{file}'
        data, file_name = load_json_data(data_path)
        if env_time:
            callback.append([str(amount), "coordinates", "load_coordinate_data", "load_json_data", perf_counter() - time])
            time = perf_counter()
        if "can" in file_name:
            labels.append(0)
        elif "peace" in file_name:
            labels.append(1)
        elif "thumb" in file_name:
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence_coordinates(data)
        if env_time:
            callback.append([str(amount), "coordinates", "load_coordinate_data", "transform_data_to_sequence_coordinates", perf_counter() - time])
            time = perf_counter()
        sequences.append(result)
    labels = np.array(labels)
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "np.array(labels)", perf_counter() - time])
        time = perf_counter()
    max_seq_length = max(len(seq) for seq in sequences)
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "max_seq_length", perf_counter() - time])
        time = perf_counter()
    print(f"Max sequence length: {max_seq_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "pad_sequences", perf_counter() - time])
        time = perf_counter()

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 63)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "model=Sequential", perf_counter() - time])
        time = perf_counter()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "model.compile", perf_counter() - time])
        time = perf_counter()

    # Train the model
    history = model.fit(padded_sequences, labels, epochs=10, batch_size=1)
    if env_time:
        callback.append([str(amount), "coordinates", "load_coordinate_data", "model.fit", perf_counter() - time])
        time = perf_counter()
    if metrics:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        print(f"Final Loss: {loss[-1]}, Final Accuracy: {accuracy[-1]}")
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predicted_labels, average='weighted')
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted')
        if env_save:
            print("Saving model metrics")
            with open("model_metrics.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"coordinates{amount}", loss[-1], accuracy[-1], f1, precision, recall])

    if save:
        file_name = f"coordinates_{amount}.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


def load_graph_data(save: bool = False, amount: int = 25, metrics: bool = False, callback: list = None):
    env_save = True if os.getenv("model_metrics") == "True" else False
    env_time = True if os.getenv("time_metrics") == "True" else False
    time = None
    if env_time:
        time = perf_counter()
    files: list = os.listdir(f'./data/final/graphs/{amount}/')
    if env_time:
        callback.append([str(amount), "graphs", "load_graph_data", "os.listdir", perf_counter() - time])
        time = perf_counter()
    print(f"Amount of files: {len(files)}")
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = f'./data/final/graphs/{amount}/{file}'
        data, file_name = load_json_data(data_path)
        if env_time:
            callback.append([str(amount), "graphs", "load_graph_data", "load_json_data", perf_counter() - time])
            time = perf_counter()
        if "can" in file_name:
            labels.append(0)
        elif "peace" in file_name:
            labels.append(1)
        elif "thumb" in file_name:
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence_graphs(data)
        if env_time:
            callback.append([str(amount), "graphs", "load_graph_data", "transform_data_to_sequence_graphs", perf_counter() - time])
            time = perf_counter()
        sequences.append(result)
    labels = np.array(labels)
    if env_time:
        callback.append(
            [str(amount), "graphs", "load_graph_data", "np.array(labels)", perf_counter() - time])
        time = perf_counter()
    max_seq_length = max(len(seq) for seq in sequences)
    if env_time:
        callback.append(
            [str(amount), "graphs", "load_graph_data", "max_seq_length", perf_counter() - time])
        time = perf_counter()
    print(f"Max sequence length: {max_seq_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")
    if env_time:
        callback.append(
            [str(amount), "graphs", "load_graph_data", "pad_sequences", perf_counter() - time])
        time = perf_counter()

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 21)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])
    if env_time:
        callback.append(
            [str(amount), "graphs", "load_graph_data", "model=Sequential", perf_counter() - time])
        time = perf_counter()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if env_time:
        callback.append(
            [str(amount), "graphs", "load_graph_data", "model.compile", perf_counter() - time])
        time = perf_counter()

    # Train the model
    history = model.fit(padded_sequences, labels, epochs=10, batch_size=1)
    if env_time:
        callback.append(
            [str(amount), "graphs", "load_graph_data", "model.fit", perf_counter() - time])
    if metrics:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predicted_labels, average='weighted')
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted')
        if env_save:
            print("Saving model metrics")
            with open("model_metrics.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"graphs{amount}", loss[-1], accuracy[-1], f1, precision, recall])

    if save:
        file_name = f"graphs_{amount}.keras"
        print("Model training complete.")
        save_model(model, filepath=file_name)


def load_combined_data(save: bool = False, amount: int = 25, metrics: bool = False, callback: list = None):
    env_save = True if os.getenv("model_metrics") == "True" else False
    env_time = True if os.getenv("time_metrics") == "True" else False
    time = None
    if env_time:
        time = perf_counter()
    files: list = os.listdir(f'./data/final/combined/{amount}')  # change after testing
    if env_time:
        callback.append([str(amount), "combined", "load_combined_data", "os.listdir", perf_counter() - time])
        time = perf_counter()
    print(f"Amount of files: {len(files)}")
    sequences: list = []
    labels: list = []
    for file in files:
        data_path = f'./data/final/combined/{amount}/{file}'
        data, file_name = load_json_data(data_path)
        if env_time:
            callback.append([str(amount), "combined", "load_combined_data", "load_json_data", perf_counter() - time])
            time = perf_counter()
        if "can" in file_name:
            labels.append(0)
        elif "peace" in file_name:
            labels.append(1)
        elif "thumb" in file_name:
            labels.append(2)
        else:
            raise ValueError("Invalid label")
        result = transform_data_to_sequence_combine(data)
        if env_time:
            callback.append([str(amount), "combined", "load_combined_data", "transform_data_to_sequence_combine", perf_counter() - time])
            time = perf_counter()
        sequences.append(result)
    labels = np.array(labels)
    if env_time:
        callback.append(
            [str(amount), "combined", "load_combined_data", "np.array(labels)", perf_counter() - time])
        time = perf_counter()
    max_seq_length = max(len(seq) for seq in sequences)
    if env_time:
        callback.append(
            [str(amount), "combined", "load_combined_data", "max_seq_length", perf_counter() - time])
        time = perf_counter()
    print(f"Max sequence length: {max_seq_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")
    if env_time:
        callback.append(
            [str(amount), "combined", "load_combined_data", "pad_sequences", perf_counter() - time])
        time = perf_counter()

    model = Sequential([
        LSTM(128, input_shape=(max_seq_length, 84)),
        Dense(3, activation='softmax')  # 3 output classes: Fish, Cow, Moon
    ])
    if env_time:
        callback.append(
            [str(amount), "combined", "load_combined_data", "model=Sequential", perf_counter() - time])
        time = perf_counter()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if env_time:
        callback.append(
            [str(amount), "combined", "load_combined_data", "model.compile", perf_counter() - time])
        time = perf_counter()

    # Train the model
    history = model.fit(padded_sequences, labels, epochs=10, batch_size=1)
    if env_time:
        callback.append(
            [str(amount), "combined", "load_combined_data", "model.fit", perf_counter() - time])
        time = perf_counter()
    if metrics:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        print(f"Final Loss: {loss[-1]}, Final Accuracy: {accuracy[-1]}")
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predicted_labels, average='weighted')
        precision = precision_score(labels, predicted_labels, average='weighted')
        recall = recall_score(labels, predicted_labels, average='weighted')
        if env_save:
            print("Saving model metrics")
            with open("model_metrics.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"combined{amount}", loss[-1], accuracy[-1], f1, precision, recall])

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


def test_model_prediction(prefix: str = '', suffix: str = '', callback: list = None):
    callback.append("test_model_prediction")
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
        return file_name, class_labels[predicted_labels[0]]


def test_data_prediction():
    path_prefix = f'./data/final/testing/'
    suffixes = ['25', '50', '100']
    models = ["coordinates", "graphs", "combined"]
    results = []
    for model in models:
        for suffix in suffixes:
            model_path = f"{model}_{suffix}.keras"
            print(f"Model used: {model_path}")
            m = load_model(model_path)
            files: list = os.listdir(path_prefix)
            for file in files:
                data_path = f'{path_prefix}{file}'
                frames = extract(data_path)
                all_data: list = []
                for frame in frames:
                    all_data.append(process_mp(frame))
                file_name = data_path.split("/")[-1].split(".")[0]
                file_name = ''.join([i for i in file_name if not i.isdigit()])
                result = None
                if 'coordinates' in model_path:
                    data_dict = convert_list_data_to_dict_cords(all_data)
                    result = transform_data_to_sequence_coordinates(data_dict)
                elif 'graphs' in model_path:
                    data_dict = convert_list_data_to_dict_graphs(all_data)
                    result = transform_data_to_sequence_graphs(data_dict)
                elif 'combined' in model_path:
                    data_dict = convert_list_data_to_dict(all_data)
                    result = transform_data_to_sequence_combine(data_dict)
                if result is None:
                    raise ValueError("Invalid prefix")
                sequences = [result]
                i1, i2 = model_path.split("_")[0], model_path.split("_")[1].split(".")[0]
                max_seq_length = sequence_lengths[i1+i2]
                padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")
                prediction = m.predict(padded_sequences)
                predicted_labels = np.argmax(prediction, axis=1)
                class_labels = {0: "Can", 1: "Peace", 2: "Thumb"}
                results.append(
                    [model_path, file_name, class_labels[predicted_labels[0]], file_name == class_labels[predicted_labels[0]].lower()])
    return results


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
                frame_data.direct_graph
            ]
            frame_data_by_landmark[landmark].append(frame_data_list)
        dict_data[f"frame{i}"] = frame_data_by_landmark
    return dict_data


def convert_list_data_to_dict_cords(data: list):
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


def convert_list_data_to_dict_graphs(data: list):
    dict_data = {}
    for i, item in enumerate(data, start=1):
        frame_data_by_landmark = {}
        for frame_data in item:
            landmark = "Landmark" + str(frame_data.landmark + 1)
            if landmark not in frame_data_by_landmark:
                frame_data_by_landmark[landmark] = []
            frame_data_list = frame_data.direct_graph

            frame_data_by_landmark[landmark] = frame_data_list
        dict_data[f"frame{i}"] = frame_data_by_landmark
    return dict_data


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
