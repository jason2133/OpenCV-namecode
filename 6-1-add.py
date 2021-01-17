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

# STEP 2 시작

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

# STEP 3 시작
# Step 3 : Apply Perspective Transform

rect = order_points(screenCnt.reshape(4, 2) / r)
# 4행 2열로 재정렬한다.

(topLeft, topRight, bottomRight, bottomLeft) = rect

w1 = abs(bottomRight[0] - bottomLeft[0])
w2 = abs(topRight[0] - topLeft[0])
h1 = abs(topRight[1] - bottomRight[1])
h2 = abs(topLeft[1] - bottomLeft[1])

maxWidth = max([w1, w2])
maxHeight = max([h1, h2])

dst = np.float32([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]])

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

print('STEP 3 : Apply Perspective Tranform')
cv2.imshow('Warped', warped)

cv2.waitKey(0)
cv2.destroyAllWindows()

def order_points(pts):
    rect = np.zeros((4, 2), dtype = 'float32')
    s = pts.sum(axis = 1)
    # axis = 1 각 행에 대한 값을 계산한다.

    # x + y의 최솟값과 최댓값
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # y - x의 최솟값과 최댓값
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# STEP 4 : Apply Adaptive Threshold

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

print('STEP 4 : Apply Adaptive Threshold')

cv2.imshow('Original', orig)
cv2.imshow('Scanned', warped)
# cv2.imwrite('scannedImage.png', warped)

cv2.waitKey(0)
cv2.destroyAllWindows()



