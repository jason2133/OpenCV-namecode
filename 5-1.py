# 스캔한 효과를 줘서 조명의 영향 제거하기

import cv2
import numpy as np
import matplotlib.pylab as plt

# threshold 문턱, 경계
# 0은 검정색, 255는 흰색
# 이미지 이진화
# cv2.threshold cv2.THRESH_BINARY

def nothing(x):
    pass

def global_threshold():
    imgfile = 'images/document.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    # 이미지 크기 Resize

    # 이미지의 가로 픽셀수를 600으로 고정해봄.
    r = 600.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 600)
    # 세로 길이에 r을 곱해서 그 비율 유지.
    # 가로 사이즈가 600인 이미지로 리사이즈.
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    WindowName = 'Window'
    TrackbarName = 'Threshold'

    cv2.namedWindow(WindowName)
    cv2.createTrackbar(TrackbarName, WindowName, 50, 255, nothing)
    # Threshold에 따른 이진화 결과에 따라 트랙바를 만들어봄.
    # Threshold의 초기값은 50, 최댓값은 255
    # nothing : 트랙바 이동시켰을때 뭐... 아무것도 안할거니 nothing을 넣어봄.

    Threshold = np.zeros(img.shape, np.uint8)
    # 모든 것을 0으로 초기화.
    # img.shape : 이미지 픽셀의 가로 픽셀과 세로 픽셀수

    while True:
        TrackbarPos = cv2.getTrackbarPos(TrackbarName, WindowName)
        # Trackbar의 기준은 Threshold 값이 된다.
        cv2.threshold(img, TrackbarPos, 255, cv2.THRESH_BINARY, Threshold)
        # Trackbar로부터 받아온 값을 Threshold 값으로 적용한다.
        # cv2.THRESH_BINARY
        # _INV
        # cv2.THRESH_TRANC
        # cv2.THRESH_TOZEROS
        cv2.imshow(WindowName, Threshold)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
    return

if __name__ == '__main__':
    global_threshold()

# OTSU 알고리즘 쓰면 더 쉽겠구만 ㅋㅋ 이미지 이진화의 원리.

    




