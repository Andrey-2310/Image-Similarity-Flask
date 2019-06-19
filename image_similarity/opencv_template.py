import cv2
import os
from numpy import fromstring, uint8

from image_similarity.stats_writer import write_stats_to_file_and_console

from detectors.sift_detector import SiftDetector
from detectors.surf_detector import SurfDetector
from detectors.brief_detector import BriefDetector
from detectors.orb_detector import OrbDetector

# init_weight_koef = 0.7
koef_delta = 0.1

sift_detector = SiftDetector()
surf_detector = SurfDetector()
brief_detector = BriefDetector()
orb_detector = OrbDetector()

detector_map = {
    "SIFT": sift_detector,
    "SURF": surf_detector,
    "BRIEF": brief_detector,
    "ORB": orb_detector
}

# determiner = "BRIEF"

default_similarity_response = 0, 0, 0, 0


def find_closest_images(detector, weight, original_bytes):
    original = cv2.imdecode(fromstring(original_bytes, uint8), cv2.IMREAD_UNCHANGED)
    orig_kp, orig_des = detector_map.get(detector).detect_and_compute(original)
    return sorted(list(map(lambda image_path: determine_similarity((orig_kp, orig_des), image_path,
                                                            detector,
                                                            weight),
                    find_images_by_path("./static/pictures/DAM"))), key=lambda s: s[1], reverse=True)[0:6]

def determine_similarity(image_kp_des, image2_Path, determiner, weight):
    flann = cv2.FlannBasedMatcher(detector_map.get(determiner).get_index_params(), dict())
    return image2_Path, collect_statistics(flann, image_kp_des, image2_Path, determiner, weight)


def collect_statistics(flann, image_kp_des, image_2_path, determiner, weight):
    good_points, number_keypoints, kp1, kp2 = calculate_good_matches(flann, image_kp_des, image_2_path, determiner, weight)

    if number_keypoints == 0:
        print(f"ATTENTION! One of the descriptors is None")
        return 0
    percentage = round(len(good_points) / number_keypoints * 100, 2)
    print(f'Matching: {percentage}')
    return percentage


# @timeit
def calculate_good_matches(flann, image_kp_des, image_2_path, determiner, weight):
    comparable = cv2.imread(image_2_path, 0)

    (kp1, des1) = image_kp_des
    kp2, des2 = detector_map.get(determiner).detect_and_compute(comparable)
    if des1 is None or des2 is None:
        return default_similarity_response
    matches = list(filter(lambda x: len(x) == 2, flann.knnMatch(des1, des2, k=2)))
    number_keypoints = len(kp1) if len(kp1) <= len(kp2) else len(kp2)
    return get_good_points_len(matches, weight, number_keypoints), number_keypoints, kp1, kp2


def get_good_points_len(matches, weight_koef, min_of_keypoints):
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < weight_koef * n.distance:
            good_points.append(m)
    # TODO: maybe not recursive but excluding approach
    return good_points \
        if len(good_points) <= min_of_keypoints \
        else get_good_points_len(matches, weight_koef - koef_delta, min_of_keypoints)


def find_images_by_path(path):
    images = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            images.append(os.path.join(root, file))
    return images