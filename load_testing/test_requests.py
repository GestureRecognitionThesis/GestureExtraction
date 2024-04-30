import requests


def test_upload_video(subpath: str = ''):
    url = f"http://localhost:8000/model/predict_{subpath}"  # Adjust the URL to match your endpoint
    files = {'video_file': ('uploaded_video.mov', open('../app/data/videos/valid/can20.MOV', 'rb'), 'video/quicktime')}
    headers = {}  # No need to explicitly set Content-Type

    try:
        response = requests.post(url, files=files, headers=headers)
        if response.status_code == 200:
            print("Upload successful")
            print(response.json())  # Print the response content
        else:
            print(f"Upload failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    test_upload_video("combined")
