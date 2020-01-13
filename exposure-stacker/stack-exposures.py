import sys

import cv2
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('less than 2 images, exiting')
        sys.exit(0)

    images = list()
    gray_images = list()
    for i in range(1, len(sys.argv)):
        print(f'loading image {sys.argv[i]}')
        img = cv2.imread(sys.argv[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        print(f'  image dtype={img.dtype} shape={img.shape}')
        images.append(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
        # cv2.imwrite(f'{sys.argv[i]}-threshold.tif', gray)
        gray_images.append(gray)
    assert len(images) == len(gray_images)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    iterations = 500
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, termination_eps)

    warp_summed = np.copy(images[0])
    weight = 1. / len(images)
    weighted_summed = weight * np.copy(images[0])
    for i in range(1, len(gray_images)):
        print(f'warping image {i}...')
        result, warp_matrix = cv2.findTransformECC(gray_images[0], gray_images[i], warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        print(f'  correlation: {result}')
        print(f'  warp matrix=\n{warp_matrix}')
        print(f'  applying warp')
        aligned = cv2.warpAffine(images[i], warp_matrix, (warp_summed.shape[1], warp_summed.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        warp_summed = cv2.add(warp_summed, aligned)
        weighted_summed = cv2.scaleAdd(1. * aligned, weight, weighted_summed, None)
        # cv2.imwrite(f'warp-summed-{i:05d}.tif', warp_summed)
        print()
    cv2.imwrite('warp-summed.tif', warp_summed)
    cv2.imwrite('weighted-summed.tif', weighted_summed.astype(np.uint16))

    print('summing images...')
    summed = np.copy(images[0])
    for i in range(1, len(images)):
        summed = cv2.add(summed, images[i])
    print(f'summed image metadata dtype={summed.dtype} shape={summed.shape}')
    cv2.imwrite('summed.tif', summed)
