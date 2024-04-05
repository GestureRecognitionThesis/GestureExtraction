import time
from locust import HttpUser, task, between


class StatusUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_status(self):
        self.client.get("/status/")
