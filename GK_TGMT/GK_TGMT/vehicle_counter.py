import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Mở video
cap = cv2.VideoCapture("test_video.mp4")

# Vị trí line
line_y = 500
dragging = False
# Biến đếm
count = 0
crossed_ids = set()

# Class cần detect (YOLO COCO)
# car = 2, truck = 7
target_classes = [2, 7]

names = model.names
def mouse_event(event, x, y, flags, param):
    global line_y

    if event == cv2.EVENT_LBUTTONDOWN:
        line_y = y
while True:
    cv2.namedWindow("Vehicle Counter")
    cv2.setMouseCallback("Vehicle Counter", mouse_event)
    ret, frame = cap.read()
    if not ret:
        break

    # Track object
    results = model.track(frame, persist=True, conf=0.4)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            cls = int(classes[i])

            # Chỉ lấy car + truck
            if cls not in target_classes:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])
            id = int(ids[i])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            label = names[cls]

            # Vẽ box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            # Hiển thị label + ID
            cv2.putText(frame, f"{label} ID:{id}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

            # Đếm khi qua line
            if cy > line_y and id not in crossed_ids:
                count += 1
                crossed_ids.add(id)

    # Vẽ line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 2)

    # Hiển thị số lượng
    cv2.putText(frame, f"Count: {count}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255), 2)

    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()