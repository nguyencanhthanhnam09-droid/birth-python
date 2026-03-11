
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime
import cv2 

#Black and white colormap

#mang pixel values to 0-255 range (8 bits)
M = 255 #1024x768
N = 255
K = 100
# ảnh có MxN pixels 
img = np.zeros((M,N), dtype=np.uint8)
for i in range(M):
    img[i,0:100] = i
# plt.imshow(img, cmap='gray')
# plt.show()
cv2.imshow('Gray Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Color colormap
img_color = np.zeros((M,N,3), dtype=np.uint8) #Red, Green, Blue

for i in range(M):
    img_color[i,0:64,0] = i  #Red channel
    img_color[i,100:150,1] = i  #Green channel
    img_color[i,200:250,2] = i  #Blue channel
# plt.imshow(img_color)
# plt.show()
# print(img_color)
cv2.imshow('Color Map', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Tạo một bức hình có kích thước 1024x768 với các giá trị pixel ngẫu nhiên
# Random grayscale image 1024x768
img_random_gray = np.random.randint(
    0, 256, (M, N), dtype=np.uint8
)

cv2.imshow('Random Gray Image 1024x768', img_random_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Random RGB image 1024x768
img_random_color = np.random.randint(
    0, 256, (M, N, 3), dtype=np.uint8
)

cv2.imshow('Random Color Image 1024x768', img_random_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Vẽ một đường chéo, tô màu đỏ cho đường chéo đó và hiển thị bức hình
#Vẽ đường chéo thủ công bằng NumPy
# Tạo ảnh màu đen
img = np.zeros((M, N, 3), dtype=np.uint8)

# Vẽ đường chéo (từ góc trên trái → góc dưới phải)
for i in range(min(M, N)):
    img[i, i] = (0, 0, 255)  # Đỏ (BGR)

cv2.imshow("Red Diagonal Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Vẽ đường chéo bằng OpenCV
img = np.zeros((M, N, 3), dtype=np.uint8)

# Vẽ đường chéo màu đỏ
cv2.line(
    img,
    (0, 0),
    (M - 1, N - 1),
    (0, 0, 255),  # đỏ (BGR)
    thickness=2
)

cv2.imshow("Red Diagonal Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Vẽ lần lượt các chữ số La Mã I đến XII tương tự như trên một chiếc đồng hồ
# ================== KÍCH THƯỚC ẢNH ==================
width, height = 1024, 768
cx, cy = width // 2, height // 2
radius = min(width, height) // 2 - 80

roman_numbers = [
    "XII", "I", "II", "III", "IV", "V",
    "VI", "VII", "VIII", "IX", "X", "XI"
]

font = cv2.FONT_HERSHEY_SIMPLEX

# ================== VÒNG LẶP REAL-TIME ==================
while True:
    # Nền trắng
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # ================== VẼ SỐ LA MÃ ==================
    for i in range(12):
        angle = (i / 12) * 2 * math.pi - math.pi / 2
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))

        cv2.putText(
            img,
            roman_numbers[i],
            (x - 25, y + 10),
            font,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Viền đồng hồ
    cv2.circle(img, (cx, cy), radius + 25, (0, 0, 0), 2)

    # ================== LẤY THỜI GIAN HIỆN TẠI ==================
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    second = now.second

    # ================== GÓC KIM ==================
    second_angle = (second / 60) * 2 * math.pi - math.pi / 2
    minute_angle = ((minute + second / 60) / 60) * 2 * math.pi - math.pi / 2
    hour_angle = ((hour % 12 + minute / 60) / 12) * 2 * math.pi - math.pi / 2

    # ================== ĐỘ DÀI KIM ==================
    second_len = radius - 20
    minute_len = radius - 40
    hour_len = radius - 100

    # ================== TỌA ĐỘ KIM ==================
    sx = int(cx + second_len * math.cos(second_angle))
    sy = int(cy + second_len * math.sin(second_angle))

    mx = int(cx + minute_len * math.cos(minute_angle))
    my = int(cy + minute_len * math.sin(minute_angle))

    hx = int(cx + hour_len * math.cos(hour_angle))
    hy = int(cy + hour_len * math.sin(hour_angle))

    # ================== VẼ KIM ==================
    cv2.line(img, (cx, cy), (sx, sy), (0, 0, 255), 2)    # kim giây (đỏ)
    cv2.line(img, (cx, cy), (mx, my), (0, 150, 0), 4)   # kim phút (xanh)
    cv2.line(img, (cx, cy), (hx, hy), (0, 0, 0), 7)     # kim giờ (đen)

    # Tâm đồng hồ
    cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1)

    # ================== HIỂN THỊ ==================
    cv2.imshow("Roman Clock - Real Time", img)

    # Nhấn ESC để thoát
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()