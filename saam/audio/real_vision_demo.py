
# SAAMOS - Real Vision Controller Test
#
# This is my first test connecting Ari's controller
# to the actual camera and object detection model.
#
# Unlike our previous audio demo, this will not use
# simulated vision results.

from saam.core.controller import handle_command
from saam.vision.service import VisionService


def main():

    # Create the real vision service.
    # It will use the camera selected by SAAM_CAMERA.
    vision = VisionService()

    try:
        # Give myself time to position the bottle.
        input("Hold a bottle in front of the camera, then press Enter...")

        # Send a command to Ari's controller.
        # These two functions replace MockVision with
        # actual detections from the camera.
        response = handle_command(
            "Where is my water bottle?",
            detect_object=vision.detect_object,
            describe_scene=vision.describe_scene
        )

        # Show the controller's response.
        print("SAAMOS response:", response)

    finally:
        # Release the camera even if something goes wrong.
        vision.close()


if __name__ == "__main__":
    main()
