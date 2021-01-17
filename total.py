# OpenCV 및 OCR모듈 설치 유무 확인

import sys
import cv2
import pytesseract
from PIL import Image

# 윈도우에서 주석해제 (tesseract_path: tesseract설치경로 확인 후 붙여넣기)
# tesseract_path = 'C:/Program Files (x86)/Tesseract-OCR'
# pytesseract.pytesseract.tesseract_cmd = tesseract_path + '/tesseract'

print ("python:", sys.version)
print ("opencv:", cv2.__version__)
print ("pytesseract:", pytesseract.image_to_string(Image.open('images/test.png')))

# OpenCV - 이미지 읽기, 쓰기 및 표시하기 (1)

import cv2

def handle_image():
    imgfile = 'images/sample.png'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    # color로 imgfile을 읽는다

    cv2.imshow('image', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.waitKey(1)
    
if __name__ == '__main__':
    handle_image()


# OpenCV - 이미지 읽기, 쓰기 및 표시하기 (2)

import cv2

def handle_image():
    imgfile = 'images/sample.png'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # WINDOW_NORMAL은 사이즈 수정이 가능하다.

    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    # 키보드 입력값을 k에 받아준다

    # wait for ESC key to exit
    if k == 27:
        cv2.destroyAllWindows()
        # cv2.waitKey(1)
    # wait for 's' key to save and exit
    elif k == ord('s'):
        cv2.imwrite('grayImage.png', img)
        cv2.destroyAllWindows()
        # cv2.waitKey(1)
        
if __name__ == '__main__':
    handle_image()


# OpenCV - 도형 외곽 추출하기 (1)

import cv2

def contour():
    imgfile = 'images/contour.jpg'
    img = cv2.imread(imgfile)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edge = cv2.Canny(imgray, 100, 200)
    edge, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('edge', edge)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Contour', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    contour() 


# OpenCV - 도형 외곽 추출하기 (2)

import cv2

def contour_approx():
    imgfile = 'images/contour2.png'
    img = cv2.imread(imgfile)
    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edge = cv2.Canny(imgray, 100, 200)
    edge, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[0]
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    cv2.drawContours(img2, [approx], 0, (0, 255, 0), 3)
    
    cv2.imshow('Contour', img)
    cv2.imshow('Approx', img2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    contour_approx() 


# OpenCV - 투영변환 구현하기 (1)

import numpy as np
import cv2

def warp_affine():
    img = cv2.imread('images/transform.png')
    
    pts1 = np.float32([[50, 50], [200, 50], [20, 200]])
    pts2 = np.float32([[70, 100], [220, 50], [150, 250]])
    
    M = cv2.getAffineTransform(pts1, pts2)
    
    result = cv2.warpAffine(img, M, (350, 300))
    
    cv2.imshow('original', img)
    cv2.imshow('Affine Transform', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
if __name__ == '__main__':
    warp_affine()


 # OpenCV - 투영변환 구현하기 (2)

import numpy as np
import cv2

def warp_perspective():
    img = cv2.imread('images/transform.jpg')
    
    topLeft = [127, 157]
    topRight = [448, 152]
    bottomRight = [579, 526]
    bottomLeft = [54, 549]
    
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    minWidth = min([w1, w2])
    minHeight = min([h1, h2])
    
    pts2 = np.float32([[0,0], [minWidth-1,0], 
                      [minWidth-1,minHeight-1], [0,minHeight-1]])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    result = cv2.warpPerspective(img, M, (int(minWidth), int(minHeight)))
    
    cv2.imshow('original', img)
    cv2.imshow('Warp Transform', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
if __name__ == '__main__':
    warp_perspective()


# OpenCV - 스캔한 듯한 효과 주기 (1)

import numpy as np
import cv2

# Callback Function for Trackbar (but do not any work)
def nothing(x):
    pass

def global_threshold():
    imgfile = 'images/document.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    
    # Resize image
    r = 600.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 600)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    WindowName = "Window"
    TrackbarName = "Threshold"
    
    # Make Window and Trackbar
    cv2.namedWindow(WindowName)
    cv2.createTrackbar(TrackbarName, WindowName, 70, 255, nothing)
    
    # Allocate destination image
    Threshold = np.zeros(img.shape, np.uint8)
    
    # Loop for get trackbar pos and process it
    while True:
        # Get position in trackbar
        TrackbarPos = cv2.getTrackbarPos(TrackbarName, WindowName)
        # Apply threshold
        cv2.threshold(img, TrackbarPos, 255, cv2.THRESH_BINARY, Threshold)
        # Show in window
        cv2.imshow(WindowName, Threshold)
        
        # wait for ESC key to exit
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
    return

if __name__ == '__main__':
    global_threshold() 


# OpenCV - 스캔한 듯한 효과 주기 (2)

import numpy as np
import cv2

def adaptive_threshold():
    imgfile = 'images/document.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    
    # Resize image
    r = 600.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 600)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # Blur image and apply adaptive threshold
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    result_without_blur = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    result_with_blur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    cv2.imshow('Without Blur', result_without_blur)
    cv2.imshow('With Blur', result_with_blur)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
if __name__ == '__main__':
    adaptive_threshold() 


# 명함인식 구현하기 - 캡처된 이미지

import numpy as np
import cv2

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def auto_scan_image():
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    # document.jpg ~ docuemnt7.jpg
    image = cv2.imread('images/document.jpg')
    orig = image.copy()
    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    print ("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    print ("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # apply the four point transform to obtain a top-down
    # view of the original image
    rect = order_points(screenCnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    
    dst = np.float32([[0,0], [maxWidth-1,0], 
                      [maxWidth-1,maxHeight-1], [0,maxHeight-1]])
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # show the original and scanned images
    print ("STEP 3: Apply perspective transform")
    cv2.imshow("Warped", warped)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    # show the original and scanned images
    print ("STEP 4: Apply Adaptive Threshold")
    cv2.imshow("Original", orig)
    cv2.imshow("Scanned", warped)
    cv2.imwrite('scannedImage.png', warped)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
if __name__ == '__main__':
    auto_scan_image()


# 명함인식 구현하기 - 웹캠(1)

import numpy as np
import cv2

def auto_scan_image_via_webcam():
    
    try: 
        cap = cv2.VideoCapture(0)
    except:
        print ('cannot load camera!')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print ('cannot load camera!')
            break
            
        k = cv2.waitKey(10)
        if k == 27:
            break
        
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show the original image and the edge detected image
        print ("STEP 1: Edge Detection")

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            screenCnt = []

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                contourSize = cv2.contourArea(approx)
                camSize = frame.shape[0] * frame.shape[1]
                ratio = contourSize / camSize
                print (contourSize)
                print (camSize)
                print (ratio)
                
                if ratio > 0.1:
                    screenCnt = approx
                    
                break 
        
        if len(screenCnt) == 0:
            cv2.imshow("WebCam", frame)
            continue
            
        else:
            # show the contour (outline) of the piece of paper
            print ("STEP 2: Find contours of paper")

            cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow("WebCam", frame)
        
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    auto_scan_image_via_webcam()


# 명함인식 구현하기 - 웹캠(2)

import numpy as np
import cv2

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def auto_scan_image_via_webcam():
    
    try: 
        cap = cv2.VideoCapture(0)
    except:
        print ('cannot load camera!')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print ('cannot load camera!')
            break
            
        k = cv2.waitKey(10)
        if k == 27:
            break

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show the original image and the edge detected image
        print ("STEP 1: Edge Detection")

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            screenCnt = []

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                contourSize = cv2.contourArea(approx)
                camSize = frame.shape[0] * frame.shape[1]
                ratio = contourSize / camSize
                print (contourSize)
                print (camSize)
                print (ratio)
                
                if ratio > 0.1:
                    screenCnt = approx
                    
                break 
        
        if len(screenCnt) == 0:
            cv2.imshow("WebCam", frame)
            continue
            
        else:
            # show the contour (outline) of the piece of paper
            print ("STEP 2: Find contours of paper")

            cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow("WebCam", frame)
            
            # apply the four point transform to obtain a top-down
            # view of the original image
            rect = order_points(screenCnt.reshape(4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = rect

            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            maxWidth = max([w1, w2])
            maxHeight = max([h1, h2])

            dst = np.float32([[0,0], [maxWidth-1,0], 
                              [maxWidth-1,maxHeight-1], [0,maxHeight-1]])

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

            # show the original and scanned images
            print ("STEP 3: Apply perspective transform")

            # convert the warped image to grayscale, then threshold it
            # to give it that 'black and white' paper effect
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

            # show the original and scanned images
            print ("STEP 4: Apply Adaptive Threshold")

            break
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
        
    
if __name__ == '__main__':
    auto_scan_image_via_webcam()


# OCR - Tesseract

from PIL import Image
import pytesseract

def ocr_tesseract():
    image_file = 'images/scannedImage.png'
    im = Image.open(image_file)
    text = pytesseract.image_to_string(im)
    im.show()

    print (text)

if __name__ == '__main__':
    ocr_tesseract()


# OCR - Project Oxford by MS

from PIL import Image
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

def print_text(json_data):
    result = json.loads(json_data)
    for l in result['regions']:
        for w in l['lines']:
            line = []
            for r in w['words']:
                line.append(r['text'])
            print (' '.join(line))
    return

def ocr_project_oxford(headers, params, data):
    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    conn.request("POST", "/vision/v1.0/ocr?%s" % params, data, headers)
    response = conn.getresponse()
    data = response.read().decode()
    print (data + "\n")
    print_text(data)
    conn.close()
    return
    
if __name__ == '__main__':
    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': '',
    }
    params = urllib.parse.urlencode({
        # Request parameters
        'language': 'unk',
        'detectOrientation ': 'true',
    })
    data = open('images/scannedImage.png', 'rb').read()
    
    try:
        image_file = 'images/scannedImage.png'
        im = Image.open(image_file)
        im.show()
        ocr_project_oxford(headers, params, data)
    except Exception as e:
        print (e)


# 명함인식 구현하기 - 웹캠 + OCR

import numpy as np
import cv2
from PIL import Image
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

def print_text(json_data):
    result = json.loads(json_data)
    for l in result['regions']:
        for w in l['lines']:
            line = []
            for r in w['words']:
                line.append(r['text'])
            print (' '.join(line))
    return

def ocr_project_oxford(headers, params, data):
    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    conn.request("POST", "/vision/v1.0/ocr?%s" % params, data, headers)
    response = conn.getresponse()
    data = response.read().decode()
    print (data + "\n")
    print_text(data)
    conn.close()
    return

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def auto_scan_image_via_webcam():
    
    try: 
        cap = cv2.VideoCapture(0)
    except:
        print ('cannot load camera')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print ('cannot load camera!')
            break
            
        k = cv2.waitKey(10)
        if k == 27:
            break

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show the original image and the edge detected image
        # print ("STEP 1: Edge Detection")

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            screenCnt = []

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                contourSize = cv2.contourArea(approx)
                camSize = frame.shape[0] * frame.shape[1]
                ratio = contourSize / camSize
                # print (contourSize)
                # print (camSize)
                # print (ratio)
                
                if ratio > 0.1:
                    screenCnt = approx
                    
                break 
        
        if len(screenCnt) == 0:
            cv2.imshow("WebCam", frame)
            continue
            
        else:
            # show the contour (outline) of the piece of paper
            print ("STEP 2: Find contours of paper")

            cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow("WebCam", frame)
            
            # apply the four point transform to obtain a top-down
            # view of the original image
            rect = order_points(screenCnt.reshape(4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = rect

            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            maxWidth = max([w1, w2])
            maxHeight = max([h1, h2])

            dst = np.float32([[0,0], [maxWidth-1,0], 
                              [maxWidth-1,maxHeight-1], [0,maxHeight-1]])

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

            # show the original and scanned images
            print ("STEP 3: Apply perspective transform")

            # convert the warped image to grayscale, then threshold it
            # to give it that 'black and white' paper effect
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

            # show the original and scanned images
            print ("STEP 4: Apply Adaptive Threshold")

            break
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    cv2.imshow("Scanned", warped)
    cv2.imwrite('scannedImage.png', warped)
    
    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': '',
    }
    params = urllib.parse.urlencode({
        # Request parameters
        'language': 'unk',
        'detectOrientation ': 'true',
    })
    data = open('scannedImage.png', 'rb').read()
    
    try:
        image_file = 'scannedImage.png'
        ocr_project_oxford(headers, params, data)
    except Exception as e:
        print (e)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
        
    
if __name__ == '__main__':
    auto_scan_image_via_webcam()


# (참고) OpenCV - 이미지에서 텍스트 영역만 찾아내기

# 출처: http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html

import glob
import os
import random
import sys
import random
import math
import json
from collections import defaultdict

import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter


def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)/2,:] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)/2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    dilated_image = cv2.convertScaleAbs(dilated_image)
    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        _, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours


def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.
    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        #print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
                print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                        i, covered_sum, new_sum, total, remaining_frac,
                        crop_area(crop), crop_area(new_crop), area, new_area_frac,
                        f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.
    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop

    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    """
    a = im.shape[0]
    b = im.shape[1]
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    dim = (int(a * scale), int(b * scale))
    new_im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    
    return scale, new_im


def process_image(path, out_path):
    orig_im = Image.open(path)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    scale, im = downscale_image(im)

    edges = cv2.Canny(im, 100, 200)

    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges)
    borders.sort(key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -5, size=(1, 20))
    maxed_cols = rank_filter(edges, -5, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered

    contours = find_components(edges)
    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    crop = find_optimal_components_subset(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)

    crop = [int(x / scale) for x in crop]  # upscale to the original image size.

    # draw and show cropped rectangle area in the original image
    rgb_im = orig_im.convert('RGB')
    draw = ImageDraw.Draw(rgb_im)
    draw.rectangle(crop, outline='red')
    rgb_im.show()

    text_im = orig_im.crop(crop)
    text_im.show()
    text_im.save(out_path)
    print('%s -> %s' % (path, out_path))


if __name__ == '__main__':
    # path = 'images/text.jpg'
    path = 'images/scannedImage.png'
    out_path = 'croppedImage.png'
    try:
        process_image(path, out_path)
    except Exception as e:
        print('%s %s' % (path, e))

        
                

