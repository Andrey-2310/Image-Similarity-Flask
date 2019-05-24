import cv2
from image_similarity.stats_writer import write_stats_to_file_and_console

from detectors.sift_detector import SiftDetector
from detectors.surf_detector import SurfDetector
from detectors.brief_detector import BriefDetector
from detectors.orb_detector import OrbDetector

init_weight_koef = 0.7
koef_delta = 0.1

sift_detector = SiftDetector()
surf_detector = SurfDetector()
brief_detector = BriefDetector()
orb_detector = OrbDetector()

detector_map = {
    "sift": sift_detector,
    "surf": surf_detector,
    "brief": brief_detector,
    "orb": orb_detector
}

determiner = "brief"

default_similarity_response = 0, 0, 0, 0


def main():
    flann = cv2.FlannBasedMatcher(detector_map.get(determiner).get_index_params(), dict())
    for x in range(1, 12):
        for y in range(1, 12):
            collect_statistics(flann, x, y)


def collect_statistics(flann, image_1, image_2):
    good_points, number_keypoints, kp1, kp2 = calculate_good_matches(flann, image_1, image_2)

    if number_keypoints == 0:
        write_stats_to_file_and_console(
            f"ATTENTION! One of the descriptors for images {image_1}, {image_2} is None\n\n")
        return
    percentage = round(len(good_points) / number_keypoints * 100, 2)
    write_stats_to_file_and_console(f'Matching: {percentage}%\nStatus: {get_matching_status(percentage)}\n\n\n')


# @timeit
def calculate_good_matches(flann, image_1, image_2):
    original = cv2.imread(f'../pictures/brandworkz/Brandworkz-Logo-{image_1}.png', 0)
    comparable = cv2.imread(f'../pictures/brandworkz/Brandworkz-Logo-{image_2}.png', 0)

    kp1, des1 = detector_map.get(determiner).detect_and_compute(original)
    kp2, des2 = detector_map.get(determiner).detect_and_compute(comparable)
    if des1 is None or des2 is None:
        return default_similarity_response
    matches = list(filter(lambda x: len(x) == 2, flann.knnMatch(des1, des2, k=2)))
    number_keypoints = len(kp1) if len(kp1) <= len(kp2) else len(kp2)
    return get_good_points_len(matches, init_weight_koef, number_keypoints), number_keypoints, kp1, kp2


def get_good_points_len(matches, weight_koef, min_of_keypoints):
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < weight_koef * n.distance:
            good_points.append(m)
    # TODO: maybe not recursive but excluding approach
    return good_points \
        if len(good_points) <= min_of_keypoints \
        else get_good_points_len(matches, weight_koef - koef_delta, min_of_keypoints)


def get_matching_status(percentage):
    if percentage > 80:
        return 'Good'
    if percentage < 30:
        return 'Bad'
    else:
        return 'Normal'
