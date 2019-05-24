import cv2


class BriefDetector:
    def __init__(self):
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                 table_number=6,  # 12
                                 key_size=12,  # 20
                                 multi_probe_level=1)  # 2

    def detect_and_compute(self, image):
        return self.brief.compute(image, self.star.detect(image, None))

    def get_index_params(self):
        return self.index_params
