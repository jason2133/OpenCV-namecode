import numpy as np
import cv2
import matplotlib.pylab as plt

def contour_approx():
    imgfile = 'images/contour2.png'
    img = cv2.imread(imgfile)
    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(imgray, 100, 200)
    edge, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    # 꼭짓점의 개수를 줄일 contour
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    epsilon = 0.1 * cv2.arcLength(cnt, True)
    # cv2.arcLength : Contour의 둘레의 길이를 계산해달라
    # cnt : 둘레를 계산할 contour
    # True : 닫힌 곡선인지 열린 곡선인지 -> True이므로 닫힌 곡선 폐곡선
    # 0.1 * 곱한거 : 둘레 길이의 0.1로.
    # 입실론 졸라 작은 수치. 근사정확도.

    # 근사정확도는 2% ~ 5% 정도로 하면 가장 적절하다.

    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # cv2.approxPolyDP
    # 다각형을 대상으로 꼭짓점을 점점 줄여나가는 함수
    # 여기에 넣는 오차만큼을 최대한으로 해가지고 꼭짓점을 줄여나가겠다
    # epsilon이 작으면 작을수록 원본이랑 비슷해질거다.

    cv2.drawContours(img2, [approx], 0, (0, 255, 0), 3)

    cv2.imshow('Contour', img)
    cv2.imshow('Approx', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    contour_approx()

