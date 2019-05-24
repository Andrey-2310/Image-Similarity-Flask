import cv2


class SurfDetector:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        self.index_params = dict(algorithm=0, trees=5)  # algorithm = 0 is for KDTree

    def detect_and_compute(self, image):
        return self.surf.detectAndCompute(image, None)

    def get_index_params(self):
        return self.index_params
