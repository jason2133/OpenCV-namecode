# 투영변환
# 포인트를 지정하지 않고 자동으로 반듯하게 하는 방법

import cv2
import numpy as np
import matplotlib.pylab as plt

def warpPerspective():
    img = cv2.imread('images/transform.jpg')

    topLeft = [127, 157]
    topRight = [448, 152]
    bottomRight = [579, 526]
    bottomLeft = [54, 549]

    # 외곽을 구해가지고.    
    # 기존 사진에 4개의 꼭짓점을 이미 찾았다는 가정을 하고 코딩에 들어간다

    # 그럼 이 4개의 꼭짓점을 어디로 옮길것인가? 이게 중요

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    # 4개의 좌표를 가진 Numpy Array.

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    # 변환될 좌표 구하기.
    # abs : 절댓값 구하기.
    # 너비 2개와 높이 2개 구하기.

    minWidth = min([w1, w2])
    minHeight = min([h1, h2])
    # 최소너비와 최소높이를 이용한다.

    pts2 = np.float32([[0, 0], [minWidth-1, 0], [minWidth-1, minHeight-1], [0, minHeight-1]])
    # 최소너비와 최소높이 구해서 그것을 변환될 좌표에 집어넣기만 하면 된다.
    # 1을 뻄으로써 비어있는 픽셀을 만들지 않는다.

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 좌표 이동시켜라
    # getAffine - Not Perspective Correct
    # 단순히 위 아래로만 이동시켜서 원근 보정을 시키지 않는다.

    # getPerspectiveTransform - Perspective Correct
    # 원근 보정까지도 한다.

    result = cv2.warpPerspective(img, M, (int(minWidth), int(minHeight)))
    # 변환할 이미지 객체, 매트릭스, 변환될 이미지의 크기

    cv2.imshow('original', img)
    cv2.imshow('Warp Transform', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    warpPerspective()

