# 투영변환 구현하기
# 포인트를 지정해서 반듯하게 구현함.
# 특정 좌표를 지정한거지.

import cv2
import numpy as np
import matplotlib.pylab as plt

def warpAffine():
    img = cv2.imread('images/transform.png')

    # Affine 행렬의 내적

    pts1 = np.float32([[50, 50], [200, 50], [20, 200]])
    pts2 = np.float32([[70, 100], [220, 50], [150, 250]])
    # pts1에서 pts2로 이동시키겠다

    M = cv2.getAffineTransform(pts1, pts2)
    # 대상 픽셀 pts1, 이동할 위치 나타내는 픽셀 pts2

    result = cv2.warpAffine(img, M, (350, 300))
    # 변환하고자 하는 img
    # 좌표 이동 matrix M
    # 350 300 : 변환된 이미지의 크기

    cv2.imshow('original', img)
    cv2.imshow('Affine Transform', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    warpAffine()

# 이동할 픽셀 몇개를 정하고 나면
# 나머지 픽셀도 그걸 따라서 이동한다

