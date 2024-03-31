# load config
import json
with open('C:\\Users\\HARSHITA\\Desktop\\filtered_data\\food\\webcam\\roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

import cv2
import base64
import numpy as np
import requests
import time

# Construct the Roboflow Infer URL
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

# webcam interface via opencv-python
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
    }, stream=True).raw

    # Parse result image
    # Read the binary data from the response object
    image_bytes = bytearray(resp.read())
    
    # Convert the binary data to a NumPy array with the specified dtype
    image_array = np.asarray(image_bytes, dtype="uint8")
    
    # Decode the NumPy array into an image using the cv2.IMREAD_COLOR flag
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image

# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

# Release resources when finished
video.release()
cv2.destroyAllWindows()
