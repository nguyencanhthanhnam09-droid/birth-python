import cv2 as cv
import numpy as np

video_path = "bang_chuyen.mp4"   # đổi tên video cho đúng
cap = cv.VideoCapture(video_path)

line_x = 480     # vị trí đường thẳng dọc
count = 0

tracked = {}     # lưu vị trí x trước đó
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (640, 480))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 1.5)

    # ✅ VẼ ĐƯỜNG THẲNG DỌC MÀU ĐỎ
    cv.line(frame, (line_x, 0), (line_x, 480), (0, 0, 255), 2)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=40
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        for (x, y, r) in circles:
            # vẽ hình tròn
            cv.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv.circle(frame, (x, y), 2, (255, 0, 0), 3)

            matched = False
            for obj_id in tracked:
                prev_x = tracked[obj_id]
                if abs(x - prev_x) < 25:
                    matched = True
                    tracked[obj_id] = x

                    # ✅ kiểm tra đi qua đường thẳng
                    if prev_x < line_x and x >= line_x:
                        count += 1
                    break

            if not matched:
                tracked[next_id] = x
                next_id += 1

    # hiển thị số lượng
    cv.putText(
        frame,
        f"Count: {count}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv.imshow("Circle Counting", frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
