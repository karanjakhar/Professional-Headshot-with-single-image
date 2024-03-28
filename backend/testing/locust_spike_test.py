from locust import HttpUser, task, between

class SpikeTestUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Time between consecutive tasks
    host = "https://karanjakhar--stable-diffusion-kj-ip-app.modal.run"  # Base host URL

    @task
    def spike_test(self):
        file_path = "/home/karan/kj_workspace/kj_ai/Professional-Headshot-with-single-image/backend/modal_inference/akhil.png"
        
        files = {'file': open(file_path, 'rb')}
        response = self.client.post("/upload", files=files)  # Using relative path

        if response.status_code == 200:
            print("File uploaded successfully")
        else:
            print(f"File upload failed with status code: {response.status_code}")
