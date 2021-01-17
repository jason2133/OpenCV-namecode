import numpy as np
import cv2
import matplotlib.pylab as plt

def contour():
    imgfile = 'images/elonmusk_a.jpg'   
    img = cv2.imread(imgfile)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지의 색공간을 바꾸는 것
    # cvtColor : convert Color
    # BGR을 GRAY로 바꾼다. 컴퓨터의 계산값을 줄여주기 위함.

    # HSV : 특정 색깔의 영역을 추출하고 싶을 때 사용한다.
    # Hue 색상값
    # Saturation 채도
    # Value 명도

    edge = cv2.Canny(imgray, 100, 200)
    # Canny Edge Detection : 가장 인기있는 Edge 찾기 알고리즘
    # 이미지 객체, 100, 200
    # 100 : Edge 값을 찾을 때 이거보다 작으면 edge 아님
    # 200 : Edge 값을 찾을 때 이거보다 높아야만 확실히 edge 이다.
    # min과 max 사이에 있는 값은 어떻게?

    # Canny는 엣지간의 연결성을 강조한다.
    # 사이에 있다하더라도 max 위에 있는 엣지랑 연결되어 있으면 edge다.
    # 사이에 있다하더라도 max 위에 있는 엣지랑 연결 안되어 있으면 edge 아니다.
    
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contour 외곽 : 검은색 0 과 흰색 255 : 2진화된 이미지만 받아서 할 수 있다
    # 외곽을 찾는 플래그다. cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    # hierarchy 계층 구조 (등고선처럼)
    # cv2.RETR_TREE : 이들간의 관계를 어떻게 나타낼 것인가? TREE 관계로 나타낸다.
    # cv2.CHAIN_APPROX_SIMPLE : contour 정보를 꼭짓점만 반환할지, 모든 정보를 반환할지.

    cv2.imshow('edge', edge)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    # 원본 이미지 위에 contours를 그리겠다.
    # -1은 contour을 모두 그리겠다. 특정 인덱스 그리고 싶으면 특정 인덱스 적어라.
    # 0 255 0 BGR 색상 - GREEN 초록색 색상이 되겠다
    # 1 - 그릴 contour의 선의 두께가 결정된다.

    cv2.imshow('Contour', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    contour()

