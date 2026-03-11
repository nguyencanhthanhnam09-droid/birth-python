import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



img = np.random.randint(0, 256, (768, 1024), dtype=np.uint8) # tạo ảnh ngẫu nhiên 1024x768 với giá trị pixel từ 0 đến 255
cv=imshow('random image', img)
key = cv.waitkey(1)
if key == ord('q'):
    cv.destroyAllWindows()