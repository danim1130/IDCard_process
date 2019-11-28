import cv2
import numpy as np
from id_scripts.id_card_models import IDCardConfiguration


def __get_transform_sift_for_type(input_img, card_config: IDCardConfiguration, target_width=640):
    min_match_count = 5

    img_height, img_width = input_img.shape[0:2]
    scale = (target_width / img_width)
    img2 = cv2.resize(input_img, (target_width, int(img_height * scale)), interpolation=cv2.INTER_LANCZOS4)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    template_img = card_config.template
    kp1, des1 = sift.detectAndCompute(template_img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        dst_pts /= scale
        m, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        img2 = cv2.warpPerspective(input_img, m, (template_img.shape[1], template_img.shape[0]))
        return img2

    else:
        return None


def detect_card(img, test_id_card_configuration: IDCardConfiguration):
    return __get_transform_sift_for_type(img, test_id_card_configuration)
