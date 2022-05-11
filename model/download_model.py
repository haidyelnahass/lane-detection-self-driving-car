# download from url
import requests

URL = "https://pjreddie.com/media/files/yolov3.weights"
response = requests.get(URL)
dest ="model/yolo.weights"
with open(dest, "wb") as file_:
    file_.write(response.content)