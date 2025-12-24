import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
print(f"Downloading from {url}...")
try:
    urllib.request.urlretrieve(url, "pose_landmarker.task")
    print("Download successful!")
except Exception as e:
    print(f"Error: {e}")
