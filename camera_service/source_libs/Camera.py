from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows, imwrite
import cv2
from numpy import ndarray
from os import mkdir
from os.path import dirname, exists


class Camera:
    cam_index: int = -1
    settings: list[(int, int)] = list()

    @staticmethod
    def __get_available_cameras() -> list[(int, VideoCapture)]:
        port: int = 0
        not_working = 0
        working_cameras: list[(int, VideoCapture)] = []
        while not_working < 6:
            camera = VideoCapture(port)
            if not camera.isOpened():
                not_working += 1
            else:
                is_reading, image = camera.read()
                if is_reading:
                    working_cameras.append((port, camera))
                else:
                    not_working += 1
            port += 1
        return working_cameras

    def select_camera(self, take_test_image: bool = False) -> None:
        available_cameras: list[(int, VideoCapture)] = Camera.__get_available_cameras()
        if len(available_cameras) == 0:
            raise Exception("No cameras available!")
        elif len(available_cameras) > 1:
            print("Available Cameras:")
            for port, cam in available_cameras:
                print(f"  {port} -- {cam.getBackendName()}")
            select_cam_id = input("Please enter the ID\n>> ")
            selected_cam_id = int(select_cam_id)
        else:
            selected_cam_id, _ = available_cameras[0]

        if take_test_image:
            cam = VideoCapture(selected_cam_id)
            success, image = cam.read()
            cam.release()
            print("Press any key to continue")
            imshow("Test Image", image)
            waitKey(0)
            destroyAllWindows()

        self.cam_index = selected_cam_id

    def take_image(self, show_image: bool = False) -> ndarray:
        cam = VideoCapture(self.cam_index)
        for setting, value in self.settings:
            cam.set(setting, value)
        success, image = cam.read()
        cam.release()
        if success:
            if show_image:
                print("Press any key to continue")
                imshow("IMG", image)
                waitKey(0)
                destroyAllWindows()
            return image
        else:
            raise Exception("Problem taking Image")

    @staticmethod
    def save_image(path: str, image: ndarray) -> bool:
        if not exists(dirname(path)):
            mkdir(path=dirname(path))

        return imwrite(path, image)

    def add_setting(self, cam_setting, value):
        self.settings.append((cam_setting, value))


if __name__ == "__main__":
    print(cv2.CAP_PROP_FOCUS)
