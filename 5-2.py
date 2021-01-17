# 이미지를 잘개 쪼개서 Threshold를 각각 구해준다
# AdaptiveThreshold

# 스캔한 듯한 효과 주기 (2)

import cv2
import numpy as np
import matplotlib.pylab as plt

def adaptive_threshold():
    imgfile = 'images/document.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    r = 600.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 600)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # 가우시안 블러
    # 이미지 리사이즈를 위에서 했고, 가우시안 블러를 통해 이미지에 블러 효과를 준다.
    # 주변 픽셀의 평균값을 대입하는 방법
    # (5, 5) : 블러 효과를 얼만큼 줄것인가? 블러 효과를 주는 주변 픽셀의 크기.

    result_without_blur = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    # 블러 처리 안한거
    # 21 Threshold 뭉텅이?
    # 10 가감 상수
    
    result_with_blur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    # 블러 처리한거
    
    cv2.imshow('Without Blur', result_without_blur)
    cv2.imshow('With Blur', result_with_blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    adaptive_threshold()

