from training import load_coordinate_data, load_graph_data, load_combined_data, test_model_prediction


def run(arg: str = ''):
    if arg == '':
        raise ValueError("Invalid argument")
    print(f"Running with argument: {arg}")
    amount = int(arg.split()[0])
    metrics = True if 'metrics' in arg else False
    saving = True if 'save' in arg else False
    if 'load' in arg:
        if 'coordinates' in arg:
            load_coordinate_data(save=saving, amount=amount, metrics=metrics)
        elif 'graphs' in arg:
            load_graph_data(save=saving, amount=amount, metrics=metrics)
        elif 'combined' in arg:
            load_combined_data(save=saving, amount=amount, metrics=metrics)
    elif 'predict' in arg:
        prefix = 'coordinates' if 'coordinates' in arg else 'graphs' if 'graphs' in arg else 'combined'
        test_model_prediction(prefix, str(amount))


if __name__ == '__main__':
    run(arg="100 predict coordinates")
