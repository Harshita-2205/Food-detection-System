import json
import asyncio
import cv2
import base64
import numpy as np
import httpx

# Load configuration from JSON file
def load_config():
    with open('roboflow_config.json', 'r') as f:
        config = json.load(f)
    return config

# Construct the Roboflow Infer URL
def get_upload_url(config):
    upload_url = "".join([
        "https://detect.roboflow.com/",
        config["ROBOFLOW_MODEL"],
        "?api_key=",
        config["ROBOFLOW_API_KEY"],
        "&format=image",  # Change to json if you want the prediction boxes, not the visualization
        "&stroke=5"
    ])
    return upload_url

# Infer via the Roboflow Infer API and return the result
async def infer(requests, config):
    try:
        # Get the current image from the webcam
        ret, img = video.read()

        # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
        height, width, channels = img.shape
        scale = config["ROBOFLOW_SIZE"] / max(height, width)
        img = cv2.resize(img, (round(scale * width), round(scale * height)))

        # Encode image to base64 string
        retval, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()

        # Get prediction from Roboflow Infer API
        resp = await requests.post(
            get_upload_url(config),
            data=img_str,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )

        # Parse result image
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image

    except cv2.error as e:
        print(f"Error encoding image: {e}")
        return None

    except httpx.HTTPError as e:
        print(f"Error making HTTP request: {e}")
        return None

# Main loop; infers at FRAMERATE frames per second until you press "q"
async def main(config):
    # Initialize
    last_frame = time.time()

    # Initialize a buffer of images
    futures = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as requests:
            while True:
                # On "q" keypress, exit
                if cv2.waitKey(1) == ord('q'):
                    break

                # Throttle to FRAMERATE fps and print actual frames per second achieved
                elapsed = time.time() - last_frame
                await asyncio.sleep(max(0, 1/config["FRAMERATE"] - elapsed))
                print(f"{1/(time.time()-last_frame)} fps")
                last_frame = time.time()

                # Enqueue the inference request and safe it to our buffer
                task = asyncio.create_task(infer(requests, config))
                futures.append(task)

                # Wait until our buffer is big enough before we start displaying results
                if len(futures) < config["BUFFER"] * config["FRAMERATE"]:
                    continue

                # Remove the first image from our buffer
                # wait for it to finish loading (if necessary)
                image = await futures.pop(0)
                # And display the inference results
                if image is not None:
                    cv2.imshow('image', image)

    except cv2.error as e:
        print(f"Error displaying image: {e}")

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        # Release resources when finished
        video.release()
        cv2.destroyAllWindows()

#