import cv2


class OrbDetector:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                 table_number=6,  # 12
                                 key_size=12,  # 20
                                 multi_probe_level=1)  # 2

    def detect_and_compute(self, image):
        return self.orb.detectAndCompute(image, None)

    def get_index_params(self):
        return self.index_params
