import cv2 as cv

src = cv.imread('cow.jpg')
smooth = cv.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
details = src - smooth
cv.imshow('smooth', smooth)
cv.imshow('details', details)
cv.waitKey(0)
