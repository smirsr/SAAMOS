import cv2 as cv

class CameraSystem:
    def __init__(self, camera):
        self.CameraNumber = camera
        self.Camera = 0

    def intialize_camera(self):
        self.Camera = cv.VideoCapture(self.CameraNumber)

    def get_frame(self):
        ret, frame = self.Camera.read()
        true_frame = cv.flip(frame, 1)
        cv.imshow("Frame", true_frame)

        if cv.waitKey(0) & 0xFF == ord('n'):
            return

        return (ret, frame)

    def destroy_camera(self):
        self.Camera.release()
        cv.destroyAllWindows()