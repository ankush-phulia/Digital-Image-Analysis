import sys
import argparse
import cv2 as cv
import numpy as np

from scipy.signal import argrelextrema

# global variables
gMinThold = 15
gMaxThold = 250


def GetParser():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('path', help='Image path')
    return Parser


def detectSkin(image):
    '''
    From pyimagesearch
    '''
    # skin hsv pixel limits
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # convert rgb to hsv - hue, saturation, value
    converted = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    skin_mask = cv.inRange(converted, lower, upper)

    # use elliptical kernel for erosions/dilations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv.dilate(skin_mask, kernel, iterations=2)

    # blur and remove noise
    skin_mask = cv.GaussianBlur(skin_mask, (3, 3), 0)
    return skin_mask


def detectFaces(image):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    return faces


def Filter(image):
    base = cv.edgePreservingFilter(image, flags=1)
    detail = image - base
    return base, detail


def BiModal(histogram, select_basis='intensity', ptage_Thold=0.05, nbd=5):
    maximae = argrelextrema(histogram, np.greater, order=10)[0]
    minimae = argrelextrema(histogram, np.less, order=10)[0]
    total_px = histogram.sum()
    b, d, m = None, None, None
    if len(maximae) > 1 and len(minimae) > 0:  # bimodal or more
        if select_basis == 'intensity':
            # take darkest maximae
            for maxima in maximae:
                if histogram[maxima - 5: maxima +
                             5].sum() > ptage_Thold * total_px:
                    d = maxima
                    break
            # take lightest maxima
            reversed_maximae = maximae[::-1]
            for maxima in reversed_maximae:
                if histogram[maxima - 5: maxima +
                             5].sum() > ptage_Thold * total_px:
                    b = maxima
                    break
            # sanity check - b != d
            if b == d:
                return None
            # get minima
            minval = gMaxThold
            for minima in minimae:
                if d < minima and histogram[minima] < 0.8 * \
                        histogram[d] and b > minima and histogram[minima] < 0.8 * histogram[b]:
                    if histogram[minima] < minval:
                        m, minval = minima, histogram[minima]

    if b and d and m:
        return ((b - d) * 1.0) / (m - d), m
    else:
        return None


def SidelightCorrect(x, y, w, h, skin_c, Iout, smooth_ksize=30, smooth_sigma=10):
    '''
    Correct Sidelit faces
    '''
    hist = cv.calcHist(
        skin_c, [2], None, [gMaxThold], [
            gMinThold, gMaxThold]).T.ravel()
    hist = np.correlate(
        hist,
        cv.getGaussianKernel(smooth_ksize, smooth_sigma).ravel(),
        'same')
    f, m = BiModal(hist)
    # mask for skin pixels < thold
    mask = (gMinThold < skin_c[:, :, 2]) & (skin_c[:, :, 2] < m)
    return np.uint8(Iout[y:y + h, x:x + h][mask] * f), mask


def faceEnhance(image):
    '''
    Implement side-lit correction and underexposure
    '''
    # get face, skin and base & details
    faces = detectFaces(image)
    skin_mask = detectSkin(image)
    base, detail = Filter(image)
    cv.imshow('tets', base + detail)
    Iout = cv.cvtColor(base, cv.COLOR_BGR2HSV)
    # cv.imshow('base', base); cv.imshow('detail', detail)

    for face in faces:
        # get the actual face
        x, y, w, h = face
        face_area = image[y:y + h, x:x + w]

        # skin part of the face
        skin = cv.bitwise_and(face_area, face_area,
                              mask=skin_mask[y:y + h, x:x + w])
        # cv.imshow('skin', skin)
        skin_c = cv.cvtColor(skin, cv.COLOR_BGR2HSV)

        # side-lit face
        try:
            value, mask = SidelightCorrect(x, y, w, h, skin_c, Iout)
            Iout[y:y + h, x:x + h][mask] = value
        except:
            print 'Not Sidelit'

        cv.imshow('sidelit', cv.cvtColor(Iout, cv.COLOR_HSV2BGR) + detail)
        cv.waitKey(0)


def skinFace(image, skin_mask):
    '''
    Get skin for faces
    '''
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        sub_image = image[y:y + h, x:x + w]
        image[y:y + h,
              x:x + w] = cv.bitwise_and(sub_image,
                                        sub_image,
                                        mask=skin_mask[y:y + h,
                                                       x:x + w])
    return image


def Run(**kwargs):
    # read input
    image = cv.imread(kwargs['path'])

    # process image(s)
    enhanced_image = faceEnhance(image)

    # display results
    # cv.imshow('Original image', image)
    # cv.imshow('Enhanced image', enhanced_image)
    # cv.waitKey(0)


if __name__ == '__main__':
    args = GetParser().parse_args(sys.argv[1:])
    Run(**vars(args))
