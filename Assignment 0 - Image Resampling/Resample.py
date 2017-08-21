import sys
import argparse
import cv2 as cv
import numpy as np


def GetParser():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('path', help='Image path')
    Parser.add_argument(
        'scale_x',
        type=float,
        help='Times up/downscale along x axis')
    Parser.add_argument(
        'scale_y',
        type=float,
        help='Times up/downscale along y axis')
    return Parser


def libScale(image, scale_x, scale_y):
    return cv.resize(image, (0, 0), fx=scale_x, fy=scale_y)


def bilinearInterpolate(image, x, y):
    x1, y1 = int(x), int(y)
    x1f, y1f = x - x1, y - y1
    x2 = min(x1 + 1, image.shape[0] - 1)
    y2 = min(y1 + 1, image.shape[1] - 1)

    # Get all the points
    p11, p12 = image[x1, y1], image[x1, y2]
    p21, p22 = image[x2, y1], image[x2, y2]

    # Interpolate
    u = y1f * p12 + (1.0 - y1f) * p11
    v = y1f * p22 + (1.0 - y1f) * p21
    return x1f * v + (1.0 - x1f) * u


def manualScale(image, scale_x, scale_y):
    x_in, y_in = image.shape[0], image.shape[1]
    x_out, y_out = int(x_in * scale_x), int(y_in * scale_y)
    scaled_image = np.empty((x_out, y_out, image.shape[2]), dtype=np.uint8)
    inv_scale_x, inv_scale_y = float(
        x_in) / float(x_out), float(y_in) / float(y_out)

    for i in xrange(x_out):
        for j in xrange(y_out):
            x, y = i * inv_scale_x, j * inv_scale_y
            scaled_image[i, j] = bilinearInterpolate(image, x, y)
    return scaled_image


def pixelDiffError(libgen_image, manual_image):
    lib_img = np.array(libgen_image, dtype=int)
    man_img = np.array(manual_image, dtype=int)
    return np.linalg.norm(lib_img - man_img)


def Run(**kwargs):
    # read input and select ROI
    image = cv.imread(kwargs['path'])
    # roi = cv.selectROI('Choose Area to Zoom', image, fromCenter=False)
    # image = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    # process image(s)
    libscaled_image = libScale(
        image, kwargs['scale_x'], kwargs['scale_y'])
    manual_scaled_image = manualScale(
        image, kwargs['scale_x'], kwargs['scale_y'])

    # display results
    cv.imshow('OpenCV scaled image', libscaled_image)
    cv.imshow('Manually scaled image', manual_scaled_image)
    cv.waitKey(0)

    print pixelDiffError(libscaled_image, manual_scaled_image)


if __name__ == '__main__':
    args = GetParser().parse_args(sys.argv[1:])
    Run(**vars(args))
