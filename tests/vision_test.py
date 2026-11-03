import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from saam.vision import CameraSystem

def test_vision():

    VisionSystem = CameraSystem(0)

    VisionSystem.intialize_camera()    
    VisionSystem.get_frame()

    VisionSystem.destroy_camera()

if __name__ == "__main__":
    test_vision()