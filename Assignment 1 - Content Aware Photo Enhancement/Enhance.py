import sys
import argparse
import cv2 as cv
import numpy as np


def GetParser():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('path', help='Image path')
    return Parser


def detectSkin(image):
    '''
    From pyimagesearch
    '''
    # skin hsv pixel limits
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    
    # convert rgb to hsv - hue, saturation, value
    converted = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    skin_mask = cv.inRange(converted, lower, upper)

    # use elliptical kernel for erosions/dilations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv.dilate(skin_mask, kernel, iterations = 2)

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


def faceEnhance(image):
    '''
    Implement side-lit correction and underexposure
    '''
    # get face, skin and base & details
    faces = detectFaces(image)
    skin_mask = detectSkin(image)
    base, detail = Filter(image)

    for face in faces:
        # get the actual face
        x, y , w, h = face
        face_area = image[y:y+h, x:x+w]
        
        # skin part of the face
        skin = cv.bitwise_and(face_area, face_area, mask=skin_mask[y:y+h, x:x+w])
        skin_c = cv.cvtColor(skin, cv.COLOR_BGR2HSV)

        # histogram of intensities
        hist = cv.calcHist(skin_c, [2], None, [200], [1, 200])
        
        from matplotlib import pyplot as plt
        plt.plot(hist)
        plt.show() 
        cv.waitKey(0)


def skinFace(image, skin_mask):
    '''
    Get skin for faces
    '''
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for face in faces:
        x, y , w, h = face
        sub_image = image[y:y+h, x:x+w]
        image[y:y+h, x:x+w] = cv.bitwise_and(sub_image, sub_image, mask=skin_mask[y:y+h, x:x+w])
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
