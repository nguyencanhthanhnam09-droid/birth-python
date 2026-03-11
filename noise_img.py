import urllib.request as request
import cv2 as cv
import numpy as np


def read_image_from_github(url):
    req = request.urlopen(url) # lấy dữ liệu về local
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8) #đọc dữ liệu
    img = cv.imdecode(arr, cv.IMREAD_COLOR) #decode sang ma trận pixel
    return img

def add_gauss_noise(img):
    mean = 0
    sigma = 20
    noise = np.random.normal(mean, sigma, img.shape)
    img_n = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img_n

def add_peper_noise(img, amount=0.02):
    noisy = img.copy()
    num_pixels = int(amount*img.size)
    #white 
    cords = [np.random.randint(0, i-1, num_pixels)  for i in img.shape] 
    noisy[cords[0], cords[1]] = 255 
    #black 
    cords = [np.random.randint(0, i-1, num_pixels)  for i in img.shape] 
    noisy[cords[0], cords[1]] = 0

    return noisy


def restore_img(img_noise):
    _img = cv.GaussianBlur(img_noise, (3,3), 0)
    return _img



if __name__== "__main__":
    # url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/lena.jpg"
    url ="https://image-us.24h.com.vn/upload/3-2021/images/2021-07-19…hien-Cong-dien-so-15-1-1626671550-857-width1200height800.jpg"
    # print(read_image_from_github(url))
    img = read_image_from_github(url)
  
    img_bw = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ed1 = cv.Canny(img, 100,200)
    h, w = ed1.shape
    mask = np.zeros_like(ed1)
    poly = np.array(
         [[ (0,h), 
        (w,h),
        (w//2+50, h//2),
        (w//2 - 50, h//2)
        ]], dtype=np.int32
)
    cv.fillPoly(mask, poly, 255)
    roi = cv.bitwise_and(ed1, mask)
    img2 = np.concatenate((ed1, roi), axis=1)
    cv.imshow("ROI", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    lines = cv.HoughLinesP(roi,
                           rho=1.0,
                           theta=np.pi/180, 
                           threshold=50,
                           minLineLength=50,
                           maxLineGap=150)
    lane_img = img.copy()
    if lines is not None:
        for line in lines:
            x1,y1, x2, y2 = line[0]
            cv.line(lane_img, (x1,y1), (x2,y2), (0, 0, 255 ),2)
    cv.imshow("img2", lane_img)
    cv.waitKey(0)
    cv.destroyAllWindows()



    # ed1 = cv.Canny(img, 100,200)
    # ed2 = cv.Canny(img_bw, 100, 200)
    # cbimg = np.concatenate((ed1, ed2), axis=1)
    # cv.imshow("edge",cbimg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # cv.imshow("perfect condition",img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # edge_img1 = cv.Canny(img, 100, 200)
    # cv.imshow("edge detection",edge_img1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # # img2 = add_gauss_noise(img)
    # img3 = add_peper_noise(img) # ảnh muối tiêu
    
    # cv.imshow("Real",img3)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # edge_img2 = cv.Canny(img3, 100, 200)
    # cv.imshow("edge detection",edge_img2)
# cv.waitKey(0)
    # cv.destroyAllWindows()

    # md_blur = cv.medianBlur(img3, 5)
    # edge_img3 = cv.Canny(md_blur, 100, 200)
    # cv.imshow("median Blur",edge_img3)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    # gauss_blur = cv.GaussianBlur(img3, (3,3),0)
    # edge_img4 = cv.Canny(gauss_blur, 100, 200)
    # cv.imshow("gauss Blur",edge_img4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    # n_blur = cv.blur(img3, (5,5))
    # edge_img5 = cv.Canny(n_blur, 100, 200)
    # cv.imshow("blur",edge_img5)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # cb_img = np.concatenate((edge_img2,edge_img3,edge_img4, edge_img5), axis=1)
    # cv.imshow("edge img", cb_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    


  


    


    # combine_img = np.concatenate((img, img2, img5), axis=1)
    # cv.imshow("img",combine_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
