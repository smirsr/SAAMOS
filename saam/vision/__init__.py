
# SAAMOS - Vision Package
#
# This file makes our vision components available
# to the rest of the SAAMOS project.
#
# We're keeping the original camera and detector
# functionality while adding Marcus's new pipeline.

# Original camera and object detection components.
from .camera import CameraError, CameraSystem
from .detector import ModelLoadError, ObjectDetector
from .service import (
    VisionService,
    configure,
    describe_scene,
    detect_object,
)

# New computer vision components from Marcus.
from .detection import (
    dedupe,
    make_detection,
    matches_label,
    normalize_label,
    position_from_bbox,
)

from .pipeline import VisionPipeline, enable_console_logging
from .scanner import SceneTracker


# These are the components other Python files
# can import directly from saam.vision.
__all__ = [
    # Original vision system
    "CameraSystem",
    "CameraError",
    "ObjectDetector",
    "ModelLoadError",
    "VisionService",
    "configure",
    "detect_object",
    "describe_scene",

    # Marcus's new vision system
    "VisionPipeline",
    "SceneTracker",
    "enable_console_logging",
    "make_detection",
    "position_from_bbox",
    "dedupe",
    "normalize_label",
    "matches_label",
]
