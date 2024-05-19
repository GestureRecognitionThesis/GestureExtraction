import csv

import numpy as np
from keras.src.saving import load_model
from keras.src.utils import pad_sequences

from presentation import convert_list_data_to_dict_graphs, convert_list_data_to_dict
from training import extract, transform_data_to_sequence_coordinates, transform_data_to_sequence_combine, \
    transform_data_to_sequence_graphs, calculate_graphs
from app.training import process_mp
from time import perf_counter


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


# Structure = [model, function, time]
def time_predict(model: str, amount: int, cb: list):
    print(f"Running with model: {model} and amount: {amount}")
    video_path = "can39.MOV"
    model_name = model + "_" + str(amount) + ".keras"
    times = []
    time = perf_counter()
    m = load_model(model_name)  # functionality
    times.append([model_name, "load_model", perf_counter() - time])
    time = perf_counter()
    frames = extract(video_path)  # functionality
    times.append([model_name, "extract", perf_counter() - time])
    time = perf_counter()
    all_data: list = []
    data_dict = {}
    for frame in frames:
        all_data.append(process_mp(frame))  # functionality
    times.append([model_name, "process_mp", perf_counter() - time])
    time = perf_counter()
    if model == "coordinates":
        data_dict = convert_list_data_to_dict_cords(all_data)  # functionality
    elif model == "graphs":
        calculate_graphs(all_data)  # functionality
        times.append([model_name, "calculate_graphs", perf_counter() - time])
        time = perf_counter()
        data_dict = convert_list_data_to_dict_graphs(all_data)  # functionality
    elif model == "combined":
        calculate_graphs(all_data)  # functionality
        times.append([model_name, "calculate_graphs", perf_counter() - time])
        time = perf_counter()
        data_dict = convert_list_data_to_dict(all_data)  # functionality
    times.append([model_name, "list_to_dict", perf_counter() - time])
    time = perf_counter()
    result = None
    if "coordinates" in model_name:
        result = transform_data_to_sequence_coordinates(data_dict)  # functionality
    elif "graphs" in model_name:
        result = transform_data_to_sequence_graphs(data_dict)  # functionality
    elif "combined" in model_name:
        result = transform_data_to_sequence_combine(data_dict)  # functionality
    times.append([model_name, "transform_data", perf_counter() - time])
    time = perf_counter()
    sequences: list = [result]  # functionality
    times.append([model_name, "sequence_list", perf_counter() - time])
    time = perf_counter()
    max_seq_length = sequence_lengths["coordinates25"]  # functionality
    times.append([model_name, "sequence_length", perf_counter() - time])
    time = perf_counter()
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype="float32")  # functionality
    times.append([model_name, "pad_sequences", perf_counter() - time])
    time = perf_counter()
    prediction = m.predict(padded_sequences)  # functionality
    times.append([model_name, "predict", perf_counter() - time])
    time = perf_counter()
    predicted_labels = np.argmax(prediction, axis=1)  # functionality
    times.append([model_name, "argmax", perf_counter() - time])
    cb.extend(times)


if __name__ == '__main__':
    callback: list = []
    models = ["coordinates", "graphs", "combined"]
    suffixes = ['25', '50', '100']
    for m in models:
        for s in suffixes:
            time_predict(m, int(s), callback)
    for c in callback:
        with open("time_metrics.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(c)
