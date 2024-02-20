class FrameData:
    def __init__(self, landmark: int, hand: int, normalized: list, relative: list):
        self.landmark: int = landmark
        self.hand: int = hand
        self.normalized: list = normalized
        self.relative: list = relative
