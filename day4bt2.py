import cv2 as cv
import numpy as np

cap  = cv.VideoCapture("bang_chuyen.mp4")
count = 0 #đếm số vật thể
vat_the = []
line_x = 100 #tạo độ đường màu đỏ

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)# chuyển từ màu đen sang trắng
    gray = cv.medianBlur(gray, 5) #làm sạch nhiều

    circle = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp =1 ,
        minDist= 10,
        param1= 50,
        param2= 30,
        minRadius=5,
        maxRadius=100
    )
    if circle:
        circle = np.uint16( np.round(circle))
        for circle in circle[0, :] :
            x, y, r = circle[0], circle[1], circle[2]
            cv.circle(frame, (x,y), r, (0,0,255), 2)
    cv.imshow("frame", frame)
    if cv.waitKey(100) == ord('q'):
        break
cv.destroyAllWindows()