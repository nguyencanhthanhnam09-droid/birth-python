import cv2 as cv
import easyocr
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
IMG_PATH = 'bienso.jpg'
# Khởi tạo Reader ngoài vòng lặp để tránh tốn tài nguyên (Tắt GPU nếu không có card NVIDIA)
reader = easyocr.Reader(['en'], gpu=False)

def crop_plate(img, x, y, w, h, pad=10):
    """Cắt vùng chứa biển số và thêm khoảng đệm (padding)"""
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]

def preprocess_plate(plate_img):
    """Tiền xử lý ảnh biển số để tăng độ chính xác cho OCR"""
    if plate_img.size == 0: return None

    h, w = plate_img.shape[:2]
    # Phóng to ảnh nếu quá nhỏ (giúp nét chữ rõ hơn)
    scale = 2.0 if h < 60 else 1.0
    resize = cv.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

    # Sử dụng ngưỡng Otsu để tự động phân tách chữ và nền
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Khử nhiễu nhẹ
    denoised = cv.fastNlMeansDenoising(binary, h=10)
    return denoised

# 1. Tải ảnh
img = cv.imread(IMG_PATH)
if img is None:
    print("Không tìm thấy file ảnh!")
    exit()

img_display = img.copy() # Bản sao để vẽ kết quả lên
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Phát hiện vùng ứng viên (Localization)
# Khử nhiễu TRƯỚC khi tìm cạnh
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(blurred, 50, 200)

# Dùng Morphology để nối các nét rời rạc
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

candidate_plates = []
img_area = img.shape[0] * img.shape[1]

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w / h
    area_ratio = (w * h) / img_area

    # Lọc theo tỉ lệ biển số dài (2.0 - 6.0) hoặc biển vuông (1.0 - 1.8)
    if (1.0 < aspect_ratio < 6.0) and (0.001 < area_ratio < 0.1):
        candidate_plates.append((x, y, w, h))

# 3. Nhận diện chữ trên từng vùng ứng viên
print(f"Tìm thấy {len(candidate_plates)} vùng ứng viên...")

for (x, y, w, h) in candidate_plates:
    plate_img = crop_plate(img, x, y, w, h)
    plate_ready = preprocess_plate(plate_img)

    if plate_ready is not None:
        # OCR với danh sách ký tự cho phép
        result = reader.readtext(
            plate_ready,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            detail=1
        )

        for (bbox, text, conf) in result:
            if conf > 0.4: # Chỉ lấy kết quả có độ tin cậy trên 40%
                print(f"Biển số: {text} | Độ tin cậy: {conf:.2f}")

                # Vẽ khung và chữ lên ảnh hiển thị
                cv.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(img_display, text, (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 4. Hiển thị kết quả cuối cùng
cv.imshow("Nhan dien Bien so xe", img_display)
cv.waitKey(0)
cv.destroyAllWindows()
