import csv
import os

from training import load_coordinate_data, load_graph_data, load_combined_data, test_model_prediction, \
    test_data_prediction
from dotenv import load_dotenv

load_dotenv()


def run(arg: str = '', cb=None):
    if arg == '':
        raise ValueError("Invalid argument")
    print(f"Running with argument: {arg}")
    amount = int(arg.split()[0])
    metrics = True if 'metrics' in arg else False
    saving = True if 'save' in arg else False
    if 'load' in arg:
        if 'coordinates' in arg:
            load_coordinate_data(save=saving, amount=amount, metrics=metrics, callback=cb)
        elif 'graphs' in arg:
            load_graph_data(save=saving, amount=amount, metrics=metrics, callback=cb)
        elif 'combined' in arg:
            load_combined_data(save=saving, amount=amount, metrics=metrics, callback=cb)
    elif 'predict' in arg:
        prefix = 'coordinates' if 'coordinates' in arg else 'graphs' if 'graphs' in arg else 'combined'
        test_model_prediction(prefix, str(amount), cb)


def run_with_test_data():
    results = test_data_prediction()
    for r in results:
        print(r)


if __name__ == '__main__':
    suffixes = ['25', '50', '100']
    models = ["coordinates", "graphs", "combined"]
    for m in models:
        for s in suffixes:
            callback = []
            arg = f'{s} load {m}'
            run(arg, callback)
            print(f"Saving time metrics for model {m} with {s} samples")
            for c in callback:
                with open("time_metrics.csv", "a", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(c)

