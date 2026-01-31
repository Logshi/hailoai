"""
Shared Detection data class used by both CPU and Hailo detectors.
"""


class Detection:
    """Single detected person with bounding box and confidence."""
    __slots__ = ("x1", "y1", "x2", "y2", "confidence")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.confidence = float(confidence)

    def __repr__(self):
        return (f"Detection(x1={self.x1}, y1={self.y1}, x2={self.x2}, "
                f"y2={self.y2}, conf={self.confidence:.2f})")
