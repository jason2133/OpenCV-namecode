import numpy as np
import cv2

def handle_image():
    imgfile = 'images/elonmusk_a.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # namedWindow : 윈도우 이름을 이렇게 지정하겠습니다
    
    # 이미지를 띄울 창에 다양한 속성을 지정할 수 있다
    # 윈도우 사이즈를 조정하기 위해서

    # cv2.WINDOW_AUTOSIZE : 사이즈 조절 X
    # cv2.WINDOW_NORMAL : 사이즈 조절 O

    cv2.imshow('image', img)

    k = cv2.waitKey(0)
    # 키보드 눌렀을 때 그 반환값 받는다.
    # 키보드 입력값의 아스키 코드

    # ESC 누를시 나간다
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('grayElonMusk.jpg', img)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    handle_image()
