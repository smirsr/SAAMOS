import cv2 as cv

class CameraSystem:
    def __init__(self, camera, key):
        self.CameraNumber = camera
        self.Camera = 0
        self.Key = key

    def intialize_camera(self):
        self.Camera = cv.VideoCapture(self.CameraNumber)

    def get_frame(self):
        ret, frame = self.Camera.read()
        true_frame = cv.flip(frame, 1)

        return (ret, true_frame)

    def display_frame(self, frame):
        cv.imshow("Displaying Frame", frame)

        if cv.waitKey(0) & 0xFF == ord(self.Key):
            return

    def destroy_camera(self):
        self.Camera.release()
        cv.destroyAllWindows()