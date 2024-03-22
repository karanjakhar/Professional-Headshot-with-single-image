import requests
import concurrent.futures
import time
import uuid

file_upload_url = "https://karanjakhar--stable-diffusion-kj-ip-app.modal.run/upload"
file_path = "/home/karan/kj_workspace/kj_ai/Professional-Headshot-with-single-image/backend/modal_inference/akhil.png"

# Number of concurrent requests to send
num_requests = 100

# Function to upload a file using the curl equivalent
def upload_file(file_path, file_upload_url):
    files = {'file': open(file_path, 'rb')}
    try:
        start_time = time.time()
        response = requests.post(file_upload_url, files=files)
        end_time = time.time()
        if response.status_code == 200:
            print(f"File uploaded successfully in {end_time - start_time} seconds")
            # Generate a unique filename
            result_filename = "./result/" + str(uuid.uuid4()) + ".jpg"
            # Save the response content with the unique filename
            with open(result_filename, "wb") as f:
                f.write(response.content)
        else:
            print(f"File upload failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"File upload failed: {e}")

# Function to run the peak test
def peak_test():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        upload_futures = [executor.submit(upload_file, file_path, file_upload_url) for _ in range(num_requests)]

        for future in concurrent.futures.as_completed(upload_futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

# Run the peak test
start_time = time.time()
peak_test()
end_time = time.time()
print(f"Peak test completed in {end_time - start_time} seconds")
