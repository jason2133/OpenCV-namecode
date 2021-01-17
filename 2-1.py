import numpy as np
import cv2

def handle_image():
    imgfile = 'images/elonmusk_a.jpg'
    img1 = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    # COLOR 그대로 읽어라

    img2 = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    cv2.imshow('color', img1)
    cv2.imshow('gray', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.waitKey(1)

if __name__ == '__main__' :
    handle_image()


