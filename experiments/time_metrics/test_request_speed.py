import csv

import requests
from time import perf_counter_ns as perf_counter

url = "http://localhost:8000/model/predict_combined"  # Adjust the URL to match your endpoint
files = {'video_file': ('uploaded_video.mov', open('./can39.MOV', 'rb'), 'video/quicktime')}
headers = {}  # No need to explicitly set Content-Type


try:
    stamp = perf_counter()
    response = requests.post(url, files=files, headers=headers)
    if response.status_code == 200:
        print("Upload successful")
        result = ["combined", perf_counter()-stamp]
        with open("response_metrics.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(result)


except Exception as e:
    print(f"Error occurred: {e}")
