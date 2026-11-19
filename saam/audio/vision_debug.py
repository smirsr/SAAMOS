
# SAAMOS - Vision Integration Debug
#
# I'm testing why the continuous camera demo detects
# objects but the controller sometimes misses them.
#
# This script lets me see the camera before capturing
# a frame, so I can make sure the bottle is visible.
#
# Press SPACE to capture and analyze one frame.
# Press Q to exit without capturing anything.

import cv2

from saam.core.controller import handle_command
from saam.vision.detector import ObjectDetector
from saam.vision.camera import CameraSystem


def main():

    # Open the camera selected by SAAM_CAMERA.
    camera = CameraSystem()

    # Load the same pretrained model used by
    # our vision teammate's live detection demo.
    detector = ObjectDetector()

    try:
        camera.initialize_camera()
        detector.load()

        print("Camera preview ready!")
        print("Press SPACE to analyze a frame, or Q to quit.")

        while True:

            # Capture the next frame from the webcam.
            success, frame = camera.get_frame()

            if not success:
                print("Could not capture a frame.")
                break

            # Display the image so I can position
            # the bottle before analyzing it.
            cv2.imshow("SAAMOS Vision Debug", frame)

            # waitKey(1) checks which key was pressed.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:  # SPACE key

                # Run the actual detection model on
                # this exact camera frame.
                detections = detector.detect(frame)

                print("\nRaw detections:", detections)

                # Convert the same detections into the
                # format Ari's controller expects.
                

                # Convert detected objects into the format
                # that Ari's controller expects.
                #
                # The detector already filtered objects
                # below its confidence threshold.
                #
                # We include "detected": True because the
                # controller checks this before describing
                # an object to the user.

                scene = [
                    {
                        "object": item["object"],
                        "detected": True,
                        "position": item["position"],
                        "confidence": item["confidence"]
                    }
                    for item in detections
                ]



                # Pass the captured scene to the controller.
                #
                # We use a lambda here because handle_command()
                # expects a callable function that returns
                # a scene when requested.
                response = handle_command(
                    "What do you see?",
                    describe_scene=lambda: scene
                )

                print("Controller response:", response)
                print("Press SPACE to test another frame.")

    finally:
        # Always close the camera and preview window.
        camera.destroy_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
