import cv2
import numpy as np
import matplotlib.pylab as plt

image = cv2.imread('images/document.jpg')
orig = image.copy()

# 이미지 리사이징
r = 800.0 / image.shape[0]
dim = (int(image.shape[1] * r), 800)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# 색깔 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# 블러 처리 -> edge 검출 더 쉽게 유도

edged = cv2.Canny(gray, 75, 200)
# edge 구하기
# 75보다 작으면 edge 아님
# 200보다 크면 edge임.

print('STEP 1 : Edge Detection')

cv2.imshow('Image', image)
cv2.imshow('Edged', edged)

cv2.waitKey(0)
cv2.destroyAllWindows()




