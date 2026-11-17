"""SAAMOS vision package. Controller entry point: VisionPipeline."""
from .detection import (dedupe, make_detection, matches_label, normalize_label,
                        position_from_bbox)
from .pipeline import VisionPipeline, enable_console_logging
from .scanner import SceneTracker
 
__all__ = ["VisionPipeline", "SceneTracker", "enable_console_logging",
           "make_detection", "position_from_bbox", "dedupe",
           "normalize_label", "matches_label"]
 
