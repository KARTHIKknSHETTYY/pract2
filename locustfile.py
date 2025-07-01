from locust import HttpUser, task
class MLTest(HttpUser):
    @task
    def predict(self):
        self.client.post("/predict", json={
            "features": [5.1, 3.5, 1.4, 0.2]
        })
# locust -f p8.py --host=http://localhost:5000