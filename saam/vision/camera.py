"""Camera capture for SAAMOS. No model code lives here."""
import os
 
import cv2 as cv
 
 
class CameraError(RuntimeError):
    """Raised when the camera cannot be opened."""
 
 
def resolve_camera_source(source=None):
    """Return an int index or a device path such as '/dev/video2'.
    
    Order: explicit argument, then the SAAM_CAMERA environment variable, then 0.
    Numeric strings ("1") become ints so the env var works for both forms.
    """
    if source is None:
        source = os.environ.get("SAAM_CAMERA", 0)
    if isinstance(source, str) and source.strip().isdigit():
        return int(source)
    return source
 
 
class CameraSystem:
    def __init__(self, camera=None, key="q", mirror=False, backend=None,
                 capture_factory=None):
        # Attribute names kept from the original class.
        self.CameraNumber = resolve_camera_source(camera)
        self.Key = key
        self.Camera = None
        self.mirror = mirror              # False keeps true left/right for localization
        self.backend = backend            # e.g. cv.CAP_V4L2 on the TI board
        self._factory = capture_factory or cv.VideoCapture  # injectable for tests
 
    def initialize_camera(self):
        if self.backend is None:
            cam = self._factory(self.CameraNumber)
        else:
            cam = self._factory(self.CameraNumber, self.backend)
        if cam is None or not cam.isOpened():
            if cam is not None:
                cam.release()
            self.Camera = None
            raise CameraError(f"Could not open camera {self.CameraNumber!r}")
        self.Camera = cam
 
    intialize_camera = initialize_camera  # old misspelled name still works
 
    def get_frame(self):
        """Return (True, frame) or (False, None). Never flips a failed capture."""
        if self.Camera is None:
            return (False, None)
        ret, frame = self.Camera.read()
        if not ret or frame is None:
            return (False, None)
        if self.mirror:
            frame = cv.flip(frame, 1)
        return (True, frame)
 
    def display_frame(self, frame, delay_ms=0):
        """Show a frame. Returns True if the quit key was pressed."""
        cv.imshow("Displaying Frame", frame)
        return (cv.waitKey(delay_ms) & 0xFF) == ord(self.Key)
 
    def destroy_camera(self):
        if self.Camera is not None:
            self.Camera.release()
            self.Camera = None
        try:
            cv.destroyAllWindows()
        except cv.error:  # headless OpenCV builds (e.g. on the board)
            pass
 
