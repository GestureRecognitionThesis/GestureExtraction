from extraction import extract
from processing import process_mp, calculate_line_equation
import json


# all_data[0] is the first frame
# all_data[0][0] is the first landmark
# the length of all_data[0] is equal to the total amount of landmarks processed in the frame

def calculate_graphs(data: list):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if i == 0:
                continue
            data[i][j].direct_graph = calculate_line_equation(data[i - 1][3], data[i][3], i)

def save_to_json(path: str, data: list):
    # Initialize an empty dictionary to store the JSON data
    json_data = {}

    # Iterate over each item in the data list
    for i, item in enumerate(data, start=1):
        # Initialize a dictionary to store frame data for each landmark
        frame_data_by_landmark = {}

        # Iterate over each FrameData object in the item
        for frame_data in item:
            landmark = "Landmark"+str(frame_data.landmark+1)
            if landmark not in frame_data_by_landmark:
                frame_data_by_landmark[landmark] = []

            frame_data_list = [
                frame_data.relative[0],
                frame_data.relative[1],
                frame_data.relative[2],
                frame_data.direct_graph
            ]
            frame_data_by_landmark[landmark].append(frame_data_list)

        # Add frame data by landmark to the JSON data dictionary
        json_data[f"frame{i}"] = frame_data_by_landmark

    # Write the dictionary to a JSON file
    with open(path+"can1.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)

# list [ [ 21 landmarks in here (FrameData) ], [ 21 landmarks in here (FrameData) ], [ 21 landmarks in here (FrameData) ] ]
if __name__ == '__main__':
    video_path = '../data/videos/can1.MOV'
    data_path = './data/'
    extracted_frames = extract(video_path)
    all_data: list = []
    for frame in extracted_frames:
        all_data.append(process_mp(frame))

    calculate_graphs(all_data)

    save_to_json(data_path, all_data)
    # print("length of data: " + str(len(all_data)))
