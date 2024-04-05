import time
from locust import HttpUser, task, between


class StatusUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_status(self):
        self.client.get("/status/")

    @task
    def predict_with_model(self):
        with open('../app/data/videos/thumb1.MOV', 'rb') as f:
            self.client.post("/model/predict", files={"video_file": f})

