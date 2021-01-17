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

cv2.wiatKey(0)
cv2.destroyAllWindows()


