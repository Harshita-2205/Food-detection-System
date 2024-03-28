import json
import cv2
import base64
import numpy as np
import requests
import time

# load config
import json
with open('C:\\Users\\HARSHITA\\Desktop\\filtered_data\\food\\webcam\\roboflow_config.json') as f:
    config = json.load(f)

ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

FRAMERATE = config["FRAMERATE"]
BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True)

    # Parse result image
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Parse the JSON response
    try:
        response = json.loads(resp.text)
        label, score = max(response['objects'], key=lambda x: x['confidence'])
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse response as JSON: {e}")
        label = None
        score = None

    return image, label, score

# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if cv2.waitKey(1) == ord('q'):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image, label, score = infer()

    # Display the inference results
    cv2.imshow('image', image)

    # Print the max prediction to the console
    print(f"Max prediction: {label} (score: {score:.2f})")

    # Print frames per second
    print((1/(time.time()-start)), " fps")

# Release resources when finished
video.release()
cv2.destroyAllWindows()