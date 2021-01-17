import cv2
import numpy as np
import matplotlib.pylab as plt

# OCR 글자인식 Tesseract를 활용.

from PIL import Image
import pytesseract

# 이거 주소를 쳐줘야되네 경로를
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

def ocr_tesseract():
    image_file = 'images/elonmusk_quote.png'
    im = Image.open(image_file)
    text = pytesseract.image_to_string(im)
    im.show()

    print(text)

if __name__ == '__main__':
    ocr_tesseract()

