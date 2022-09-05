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
import networkx as nx

# Add any python libraries here

def parse():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=500,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--image_dir', default="../Data/Train/Set1",
                        help='image directory')
    Parser.add_argument('--result_dir', default="./Results/Train/Set1",
                        help='result directory')
    Parser.add_argument('--min_match_count', default=10,
                        help='minimum match count')
    args = Parser.parse_args()
    return args


def main(args):
    num_features = args.NumFeatures
    image_dir = args.image_dir
    result_dir = args.result_dir
    min_match_count = args.min_match_count

    """
    Read a set of images for Panorama stitching
    """
    images = []
    image_names = []
    images_color = []
    image_ids = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(image_path)
        image_id = image_name.split('.')[0]

        images_color.append(image_color)
        images.append(image_grey)
        image_names.append(image_id)
        image_ids.append(int(image_id))

    # images_color = [x for _,x in sorted(zip(image_ids, images_color))]
    # images = [x for _, x in sorted(zip(image_ids, images))]
    # image_names = [x for _, x in sorted(zip(image_ids, image_names))]

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

    """
    Refine: RANSAC, Estimate Homography
    """
    combs = list(itertools.combinations(range(len(images)), 2))
    homographies = [None]*len(combs)
    changed = False
    considered_combs = np.full(len(combs), False)
    computed_set = set()
    homographies = {}

    while (changed or np.any(np.logical_not(considered_combs))) and (len(computed_set) < len(images)):
        changed = False
        for index, comb in enumerate(combs):
            if considered_combs[index]:
                continue

            i, j = comb
            if computed_set:
                if (i in computed_set and j in computed_set) or (j not in computed_set and i not in computed_set):
                    continue

            considered_combs[index] = True

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

            if len(good) <= min_match_count:
                continue

            src_features = [image_feature1[m.queryIdx] for m in good]
            dst_features = [image_feature2[m.trainIdx] for m in good]

            src_pts = np.float32([image_feature1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([image_feature2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            homography_finder = FindHomography(src_pts, dst_pts, 5.0, 2000, 0.995)
            H, inlier_match, good_count = homography_finder.findHomography()

            if H is None:
                continue

            if good_count < 7:
                continue

            img4 = cv2.drawMatches(image_color1, src_features, image_color2, dst_features, inlier_match,
                                   None, **draw_params)
            image_name = 'ransac_match_' + image_id1 + '_' + image_id2 + '.png'
            image_path = os.path.join(result_dir, image_name)
            cv2.imwrite(image_path, img4)

            computed_set.add(i)
            computed_set.add(j)

            changed = True

            homographies[comb] = H
            homographies[(j, i)] = np.linalg.inv(H)

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """

    # need to define a reference frame so that we can transfer every frame to it.
    frame_count = np.zeros(len(images))
    G = nx.Graph()
    for comb in homographies:
        i, j = comb
        frame_count[i] += 1
        frame_count[j] += 1
        G.add_nodes_from(comb)
        G.add_edge(i, j)

    ref_frame = np.argmax(frame_count)
    # ref_frame = 3
    ref_frame_img = images_color[ref_frame]
    p = nx.shortest_path(G, target=ref_frame)

    homography_to_ref = {}
    print('ref frame is', ref_frame)
    print(p)
    for i in p:
        if i == ref_frame:
            continue
        path = p[i]
        H = np.eye(3)
        for j in range(len(path)-1):
            H = homographies[(path[j], path[j+1])] @ H
        homography_to_ref[(i, ref_frame)] = H


    delta_x = 0
    delta_y = 0
    for comb in homography_to_ref:
        i, j = comb

        image_color1 = images_color[i]

        H = homography_to_ref[comb]
        corners = np.asarray([[0, 0],
                              [0, image_color1.shape[0]],
                              [image_color1.shape[1], image_color1.shape[0]],
                              [image_color1.shape[1], 0]]).reshape(-1, 1, 2).astype(float)

        dst_corners = np.squeeze(cv2.perspectiveTransform(corners, H))
        bottom_left = np.amin(dst_corners, axis=0)

        if bottom_left[0] < 0:
            delta_x = max(np.ceil(np.abs(bottom_left[0])), delta_x)

        if bottom_left[1] < 0:
            delta_y = max(np.ceil(np.abs(bottom_left[1])), delta_y)

    delta_x = int(delta_x)
    delta_y = int(delta_y)

    print('delta_x is ', delta_x)
    print('delta_y is ', delta_y)

    image_width = delta_x + ref_frame_img.shape[1]
    image_height = delta_y + ref_frame_img.shape[0]

    for comb in homography_to_ref:
        i, j = comb

        image_color1 = images_color[i]

        H = homography_to_ref[comb]
        corners = np.asarray([[0, 0],
                              [0, image_color1.shape[0]],
                              [image_color1.shape[1], image_color1.shape[0]],
                              [image_color1.shape[1], 0]]).reshape(-1, 1, 2).astype(float)

        H[0, :] += H[2, :] * delta_x
        H[1, :] += H[2, :] * delta_y

        dst_corners = np.squeeze(cv2.perspectiveTransform(corners, H))
        upper_right = np.amax(dst_corners, axis=0)

        image_width = int(max(upper_right[0], image_width))
        image_height = int(max(upper_right[1], image_height))
    print(image_width, image_height)
    dst = np.zeros((image_height, image_width, 3), dtype=float)

    for comb in homography_to_ref:
        i, j = comb

        image_color1 = images_color[i]
        H = homography_to_ref[comb]

        dst1 = cv2.warpPerspective(image_color1, H, (image_width, image_height))
        index = dst1 > 0
        dst[index] = dst1[index]

    dst[delta_y:delta_y + ref_frame_img.shape[0], delta_x:delta_x + ref_frame_img.shape[1]] = ref_frame_img

    # if np.divide(image_width, image_height) > 16.0/9.0:
    #     dst = cv2.resize(dst, dsize=None, fx=3840.0 / image_width, fy=3840.0 / image_width)
    # else:
    #     dst = cv2.resize(dst, dsize=None, fx=2160.0 / image_height, fy=2160.0 / image_height)


    image_name = 'mypano.png'
    image_path = os.path.join(result_dir, image_name)
    cv2.imwrite(image_path, dst)



if __name__ == "__main__":
    args = parse()
    main(args)
