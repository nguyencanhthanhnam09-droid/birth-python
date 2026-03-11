import cv2 as cv
import numpy as np

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    base_frame = None
    while True :
        ret, frame = cap.read()
        if not ret:
            break
        if ret is not None:
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            gray = cv.GaussianBlur(gray, (25,25),0)
        if cv.waitKey(10) == ord('q'):
            break
        if base_frame is None:
            base_frame = gray
            continue
        # tìm sự khác biệt giữa 2 khung hình
        delta = cv.absdiff(base_frame, gray)
        _, nguong = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)[1]
         
        #giản nở biên 
        nguong = cv.dilate(nguong, None, iterations=2)
         
        #tìm đường biên bao phủ các vùng chuyển động
        bien, _ = cv.findContours(nguong.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
         
        for b in bien:
            if cv.contourArea(b) < 500: #nếu nhỏ quá thì bỏ qua chuyển động
                continue
            (x,y,w,h) = cv.boundingRect(b)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv.imshow("frame", frame)
        cv.imshow("Bw", nguong)
        if cv.waitKey(100) == ord('q'):
            break
    cv.destroyAllWindows()


#bài tập 2: sử dụng HoughCircles
#sử dụng video được cung cấp : viết chương trình đến xem có bao nhiêu vật theer hịnh tròn di chuyển qua line màu đỏ
# submit thông qua github