import cv2

from source_libs.Camera import Camera
from source_libs.Mqtt import MqttClient
from datetime import datetime
from os.path import abspath, dirname, join
from sys import exit
from time import sleep

camera: Camera = None
mqtt_client: MqttClient = None

BROKER_IP: str = "localhost"
BROKER_PORT: int = 1883

SLEEP_BEFORE_IMAGE_CAPTURE: float = 5.0


def setup_camera():
    global camera
    try:
        camera = Camera()
        camera.select_camera(take_test_image=False)
        camera.add_setting(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # set image to 1920x1080
        camera.add_setting(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.add_setting(cv2.CAP_PROP_AUTOFOCUS, 0)  # disable autofokus
        camera.add_setting(cv2.CAP_PROP_FOCUS, 10)
        camera.add_setting(cv2.CAP_PROP_FPS, 10)
        camera.add_setting(cv2.CAP_PROP_BACKLIGHT, 1)
    except Exception as e:
        print(e)
        exit()


def take_and_store_image():
    sleep(SLEEP_BEFORE_IMAGE_CAPTURE)
    print("Taking image...")
    try:
        image = camera.take_image(show_image=False)
        if camera.save_image(path=f"{abspath(join(dirname('.')))}/images/{datetime.now()}.png", image=image):
            print("Saved image")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    setup_camera()

    mqtt_client = MqttClient()
    mqtt_client.start(broker=BROKER_IP, port=BROKER_PORT)
    mqtt_client.subscribe(topic="eip://uni-due.de/es/enV5/DO/MEASUREMENTS", subscription_callback=take_and_store_image)
    input("Press any key to exit...\n")
    mqtt_client.stop()
