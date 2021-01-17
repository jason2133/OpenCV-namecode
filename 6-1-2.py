import cv2
import numpy as np
import matplotlib.pylab as plt

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    # contour 길이 반환

    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # 길이에 2% 정도를 오차로 한다. 도형을 근사해서 구한다.

    if len(approx) == 4:
        # 꼭짓점이 4개라면
        screenCnt = approx
        # 그것이 명함의 외곽이다.
        break

# 찾아낸 외곽 중에서 가장 큰 순서대로 꼭짓점 4개 그게 바로 명함의 외곽이다.    

print('STEP 2: Find contours of paper')
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

