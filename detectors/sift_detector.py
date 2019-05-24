import cv2


class SiftDetector:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.index_params = dict(algorithm=0, trees=5)  # algorithm = 0 is for KDTree

    def detect_and_compute(self, image):
        return self.sift.detectAndCompute(image, None)

    def get_index_params(self):
        return self.index_params

