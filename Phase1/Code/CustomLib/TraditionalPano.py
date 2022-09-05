import numpy as np
import cv2
from scipy import ndimage, misc
from scipy.linalg import null_space, svd
import os
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


class FindHomography:
    def __init__(self, points1, points2, ransacReprojThreshold=3, maxIters=2000, confidence=0.995):
        self.points1 = points1
        self.points2 = points2
        self.ransacReprojThreshold = ransacReprojThreshold
        self.maxIters = maxIters
        self.confidence = confidence
        self.modelPoints = 4

    def findInliers(self, H):
        dst_est = cv2.perspectiveTransform(self.points1, H)
        err = self.points2 - dst_est
        err = np.square(np.squeeze(err))
        err = np.sum(err, axis=1)
        inlier_mask = err <= self.ransacReprojThreshold
        return inlier_mask, np.sum(inlier_mask)

    def getSubset(self, maxAttempts=1000):
        count = len(self.points1)
        for iters in range(maxAttempts):
            idx = np.random.uniform(0, count, self.modelPoints).astype(int).tolist()
            idx_set = set(idx)
            ms1 = self.points1[idx]
            ms2 = self.points2[idx]

            while len(idx) != len(idx_set):
                idx = np.random.uniform(0, count, self.modelPoints).astype(int).tolist()
                idx_set = set(idx)
                ms1 = self.points1[idx]
                ms2 = self.points2[idx]

            # check to see if  the 4 pts lie on the same line.
            if self.checkSubset(ms1, ms2):
                return True, ms1, ms2
        return False, None, None

    def checkSubset(self, ms1, ms2):
        threshold = 0.996
        combs = {(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)}
        for inp in range(2):
            if inp:
                ptr = ms2
            else:
                ptr = ms1
            # check that i, j, k does not lie on the same line
            for i, j ,k in combs:
                d1 = ptr[j] - ptr[i]
                n1 = np.sum(np.square(d1))
                d2 = ptr[k] - ptr[i]
                denom = np.sum(np.square(d2)) * n1
                num = d1[0, 0]*d2[0, 0] + d1[0, 1]*d2[0, 1]
                if num * num > threshold * threshold * denom:
                    return False

            # We check whether the minimal set of points for the homography estimation
            # are geometrically consistent. We check if every 3 correspondences sets
            # fulfills the constraint.
            #
            # The usefulness of this constraint is explained in the paper:
            #
            # "Speeding-up homography estimation in mobile devices"
            # Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
            # Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela

            negative = 0
            for i, j, k in combs:
                A = np.asarray([[ms1[i, 0, 0], ms1[i, 0, 1], 1],
                                [ms1[j, 0, 0], ms1[j, 0, 1], 1],
                                [ms1[k, 0, 0], ms1[k, 0, 1], 1]])

                B = np.asarray([[ms2[i, 0, 0], ms2[i, 0, 1], 1],
                                [ms2[j, 0, 0], ms2[j, 0, 1], 1],
                                [ms2[k, 0, 0], ms2[k, 0, 1], 1]])
                negative += (np.linalg.det(A) * np.linalg.det(B) < 0)

            negative = int(negative)
            if negative and negative != 4:
                return False

        return True

    def run(self):
        result = False
        niters = self.maxIters
        maxGoodCount = 0
        i = 0
        count = len(self.points1)
        best_model = None
        best_mask = None
        while i < niters:
            found, ms1, ms2 = self.getSubset(1000)
            if not found:
                if not i == 0:
                    return False
                break

            H = self.runKernel(ms1, ms2)

            inlier_mask, goodCount = self.findInliers(H)

            if goodCount > max(maxGoodCount, self.modelPoints - 1):
                best_model = H.copy()
                best_mask = inlier_mask.copy()
                maxGoodCount = goodCount
                niters = self.RANSACUpdateNumIters((count - goodCount)/count, niters)

            i += 1

        if maxGoodCount:
            result = True

        return result, best_model, best_mask

    def findHomography(self):
        H = None
        result, best_model, best_mask = self.run()
        if result:
            src = self.points1[best_mask]
            dst = self.points2[best_mask]
            H = self.runKernel(src, dst)
            inlier_mask, good_count = self.findInliers(H)
            inlier_match = []
            for i in range(len(inlier_mask)):
                if inlier_mask[i]:
                    inlier_match.append(cv2.DMatch(i, i, 1))

            return H, inlier_match, good_count

    def runKernel(self, m1, m2):
        count = len(m1)

        M = np.squeeze(m1)
        m = np.squeeze(m2)

        cm_x = 0
        cm_y = 0

        cM_x = 0
        cM_y = 0

        for i in range(count):
            cm_x += m[i, 0]
            cm_y += m[i, 1]

            cM_x += M[i, 0]
            cM_y += M[i, 1]

        cm_x /= count
        cm_y /= count

        cM_x /= count
        cM_y /= count

        sm_x = 0
        sm_y = 0

        sM_x = 0
        sM_y = 0

        for i in range(count):
            sm_x += np.abs(m[i, 0] - cm_x)
            sm_y += np.abs(m[i, 1] - cm_y)
            
            sM_x += np.abs(M[i, 0] - cM_x)
            sM_y += np.abs(M[i, 1] - cM_y)

        sm_x = count/sm_x
        sm_y = count/sm_y

        sM_x = count/sM_x
        sM_y = count/sM_y

        invHnorm = np.asarray([[1./sm_x, 0, cm_x], [0, 1./sm_y, cm_y], [0, 0, 1]])
        Hnorm2 = np.asarray([[sM_x, 0, -cM_x*sM_x], [0, sM_y, -cM_y*sM_y], [0, 0, 1]])

        L = np.zeros((2*count, 9))
        for i in range(count):
            x = (m[i, 0] - cm_x)*sm_x
            y = (m[i, 1] - cm_y)*sm_y

            X = (M[i, 0] - cM_x)*sM_x
            Y = (M[i, 1] - cM_y)*sM_y

            L[2*i    , :] = [0, 0, 0, -X, -Y, -1,  X*y,  Y*y,  y]
            L[2*i + 1, :] = [X, Y, 1,  0,  0,  0, -X*x, -Y*x, -x]

        _, _, Vh = svd(L)
        H = Vh[-1, :]
        H = H.reshape((3, 3))
        Htemp = invHnorm @ H
        H = Htemp @ Hnorm2

        return H

    def RANSACUpdateNumIters(self, ep, maxIters):
        p = max(self.confidence, 0)
        p = min(p, 1.)

        ep = max(ep, 0.)
        ep = min(ep, 1.)

        num = 1 - p
        denom = 1 - np.power(1 - ep, self.modelPoints)

        num = np.log(num)
        denom = np.log(denom)

        if denom >= 0 or num <= maxIters*denom:
            pass
        else:
            maxIters = int(num/denom)

        return maxIters

def pair_wise_pano(homographies, images_color, image_names, result_dir):
    for comb in homographies:
        i, j = comb

        image_color1 = images_color[i]
        ref_frame_img = images_color[j]

        image_id1 = image_names[i]
        image_id2 = image_names[j]

        delta_x = 0
        delta_y = 0

        corners = np.asarray([[0, 0],
                             [0, image_color1.shape[0]],
                             [image_color1.shape[1], image_color1.shape[0]],
                             [image_color1.shape[1], 0]]).reshape(-1, 1, 2).astype(float)
        H = homographies[comb]

        dst_corners = np.squeeze(cv2.perspectiveTransform(corners, H))
        bottom_left = np.amin(dst_corners, axis=0)

        if bottom_left[0] < 0:
            delta_x = max(np.ceil(np.abs(bottom_left[0])), delta_x)

        if bottom_left[1] < 0:
            delta_y = max(np.ceil(np.abs(bottom_left[1])), delta_y)

        delta_x = int(delta_x)
        delta_y = int(delta_y)

        H[0, :] += H[2, :] * delta_x
        H[1, :] += H[2, :] * delta_y

        dst_corners = np.squeeze(cv2.perspectiveTransform(corners, H))
        upper_right = np.amax(dst_corners, axis=0)

        image_width = delta_x + ref_frame_img.shape[1]
        image_height = delta_y + ref_frame_img.shape[0]

        image_width = int(max(upper_right[0], image_width))
        image_height = int(max(upper_right[1], image_height))

        dst = cv2.warpPerspective(image_color1, H, (image_width, image_height))
        dst[delta_y:delta_y + ref_frame_img.shape[0], delta_x:delta_x + ref_frame_img.shape[1]] = ref_frame_img

        image_name = 'pano_' + image_id1 + '_' + image_id2 + '.png'
        image_path = os.path.join(result_dir, image_name)
        cv2.imwrite(image_path, dst)

if __name__ == '__main__':
    sigmas = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
