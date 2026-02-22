"""
Module 1: Webcam Capture
Uses OpenCV for reliable, simple webcam access.
"""

import cv2
from typing import Optional, Tuple


def open_camera(device_id: int = 0) -> cv2.VideoCapture:
    """Open default webcam. Returns VideoCapture object."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device_id}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional["cv2.Mat"]]:
    """
    Read one frame from the camera.
    Returns (success, frame). Frame is BGR numpy array or None.
    """
    ret, frame = cap.read()
    return ret, frame if ret else None


def release_camera(cap: cv2.VideoCapture) -> None:
    """Release camera resource."""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = open_camera()
    try:
        while True:
            ret, frame = read_frame(cap)
            if not ret or frame is None:
                break
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        release_camera(cap)
