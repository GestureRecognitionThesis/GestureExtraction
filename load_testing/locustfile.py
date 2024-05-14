import time
from locust import HttpUser, task, between
import random

class StatusUser(HttpUser):
    wait_time = between(1, 4)

    #@task
    #def predict_with_model(self):
     #   with open('../app/data/videos/valid/thumb1.MOV', 'rb') as f:
     #       self.client.post("/model/predict", files={"video_file": f})

    @task
    def load_test_with_random_videos(self):
        model = "coordinates"
        labels = ['thumb', 'peace', 'can']
        number_to_100 = random.randint(1, 100)
        random_label = random.choice(labels)
        with open(f'../app/data/videos/valid/{random_label}{number_to_100}.MOV', 'rb') as f:
            self.client.post(f"/model/predict_{model}", files={"video_file": f})

