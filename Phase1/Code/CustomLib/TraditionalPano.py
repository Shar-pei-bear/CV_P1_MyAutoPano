import numpy as np
import cv2
from scipy import ndimage, misc


def im_regional_max(corner_score):
    kernel = np.ones((5, 5), dtype=np.uint8)
    kernel[2, 2] = 0
    dilation = cv2.dilate(corner_score, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
    errosion = cv2.erode(corner_score, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
    # compare neighbor
    local_maxima = np.logical_and(corner_score > dilation, corner_score > errosion)

    # remove local maxima at border
    local_maxima[0, :] = False
    local_maxima[-1, :] = False
    local_maxima[:, 0] = False
    local_maxima[:, -1] = False

    return local_maxima.astype(np.uint8)


def ANMS(local_maxima, c_img, num_features):
    print(np.sum(local_maxima))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(local_maxima, connectivity=8)
    print(num_labels)
    eds = []

    # centroids = np.asarray(centroids, dtype=int)
    for i in range(1, num_labels):
        yi, xi = centroids[i]

        c_i = c_img[int(xi), int(yi)]
        r_i = np.inf
        coord = tuple(np.fliplr(centroids[1:num_labels]).astype(int).T.tolist())

        c_j = c_img[coord]
        index = c_j > c_i
        if np.any(index):
            xj = centroids[1:num_labels, 1]
            yj = centroids[1:num_labels, 0]

            xj = xj[index]
            yj = yj[index]

            ed = np.sqrt(np.square(xj - xi) + np.square(yj - yi))
            r_i = np.amin(ed)

        eds.append(r_i)

    index = np.argsort(eds)
    index = np.flip(index)
    centroids = centroids[1:num_labels]
    centroids = centroids[index]
    centroids = centroids[0:num_features]
    features = cv2.KeyPoint.convert(centroids)

    return features


def feature_extraction(img, features):
    k_size = 41
    blur = cv2.GaussianBlur(img, (k_size, k_size), 0, borderType=cv2.BORDER_CONSTANT)

    pad_width = int((k_size-1)/2)
    img_padded = np.pad(blur, pad_width)
    descriptors = []
    pts = cv2.KeyPoint.convert(features)
    for y, x in pts:
        y = int(y)
        x = int(x)
        patch = img_padded[x+pad_width:x+k_size+pad_width, y+pad_width:y+k_size+pad_width]
        descriptor = cv2.resize(patch, dsize=(8, 8)).flatten()
        descriptor = (descriptor - descriptor.mean())/np.std(descriptor)
        descriptors.append(descriptor)

    descriptors = np.asarray(descriptors)
    return descriptors


def feature_matching(image_descriptor1, image_descriptor2, num_features):
    good = []

    vk = np.full(num_features, -1)
    vl = np.full(num_features, -1)

    for k, descriptor in enumerate(image_descriptor1):
        dists = np.linalg.norm(image_descriptor2 - descriptor, axis=1)

        best_match = np.argmin(dists)
        best_match_dist = dists[best_match]
        dists[best_match] = np.inf

        second_best_match = np.argmin(dists)
        second_best_match_dist = dists[second_best_match]

        if best_match_dist < 0.7 * second_best_match_dist and vl[best_match] < 0:
            vk[k] = best_match
            vl[best_match] = k

            good.append(cv2.DMatch(k, best_match, best_match_dist))

    return good

if __name__ == '__main__':
    sigmas = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
