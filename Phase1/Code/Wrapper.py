#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import argparse
import os
import itertools
from CustomLib.TraditionalPano import *


# Add any python libraries here

def parse():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--image_dir', default="../Data/Train/Set1",
                        help='image directory')
    Parser.add_argument('--result_dir', default="./Results/Train/Set1",
                        help='result directory')
    args = Parser.parse_args()
    return args


def main(args):
    num_features = args.NumFeatures
    image_dir = args.image_dir
    result_dir = args.result_dir

    """
    Read a set of images for Panorama stitching
    """
    images = []
    image_names = []
    images_color = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(image_path)
        image_id = image_name.split('.')[0]

        images_color.append(image_color)
        images.append(image_grey)
        image_names.append(image_id)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    images_harris = []
    thresh = 110
    blockSize = 2
    apertureSize = 3
    k = 0.04

    for index in range(len(images)):
        img = images[index]
        img_color = images_color[index].copy()
        image_id = image_names[index]

        # Detecting corners
        dst = cv2.cornerHarris(img, blockSize, apertureSize, k)
        # Normalizing
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Drawing a circle around corners
        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > thresh:
                    cv2.circle(img_color, (j, i), 3, (255, 255, 255), -1)

        image_name = 'corner_' + image_id + '.png'
        image_path = os.path.join(result_dir, image_name)
        cv2.imwrite(image_path, img_color)
        images_harris.append(dst_norm)
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    image_features = []
    for index in range(len(images)):
        dst_norm = images_harris[index]
        local_maxima = im_regional_max(dst_norm)
        sorted_features = ANMS(local_maxima, dst_norm, num_features)
        image_features.append(sorted_features)
        img_color = images_color[index]
        image_id = image_names[index]
        output_image = cv2.drawKeypoints(img_color, sorted_features, 0, color=(255, 255, 255))
        image_name = 'ANMS_' + image_id + '.png'
        image_path = os.path.join(result_dir, image_name)
        cv2.imwrite(image_path, output_image)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    image_descriptors = []
    for index, img in enumerate(images):
        image_descriptors.append(feature_extraction(img, image_features[index]))

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    combs = list(itertools.combinations(range(len(images)), 2))
    homographies = [None]*len(combs)

    changed = False
    considered_combs = np.full(len(combs), False)
    computed_set = set()

    while changed or np.any(np.logical_not(considered_combs)):
        changed = False
        for index, comb in enumerate(combs):
            if considered_combs[index]:
                continue

            considered_combs[index] = True

            i, j = comb
            image_feature1 = image_features[i]
            image_feature2 = image_features[j]

            image_descriptor1 = image_descriptors[i]
            image_descriptor2 = image_descriptors[j]

            image_color1 = images_color[i]
            image_color2 = images_color[j]

            image_id1 = image_names[i]
            image_id2 = image_names[j]

            good = feature_matching(image_descriptor1, image_descriptor2, num_features)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               flags=2)
            img3 = cv2.drawMatches(image_color1, image_feature1, image_color2, image_feature2,
                                   good, None, **draw_params)
            image_name = 'match_' + image_id1 + '_' + image_id2 + '.png'
            image_path = os.path.join(result_dir, image_name)
            cv2.imwrite(image_path, img3)

            src_pts = np.float32([image_feature1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([image_feature2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            print(src_pts.shape)
            print(dst_pts.shape)


    """
    Refine: RANSAC, Estimate Homography
    """

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """


if __name__ == "__main__":
    args = parse()
    main(args)
