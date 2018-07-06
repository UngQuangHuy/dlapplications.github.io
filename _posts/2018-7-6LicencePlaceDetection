---
layout: post
title: Nhận diện bảng số xe (Phần 1)
subtitle: Deep learning ứng dụng
hidden: false
tags: [Deep learning ứng dụng]
math: true
---

# Nền tảng của deep learning - Multi-layer Perceptron
### Mục lục:
1. [Mở đầu](#intro)
2. [Phát hiện bảng số xe](#notation )
3. [Từ input tới output ](#forward)
4. [Kết bài](#conclusion)


### 1. Mở đầu <a name="intro"></a>
Hôm nay nhóm sẽ bắt đầu một số bài viết giúp bạn thực hiện một ứng dụng thực tế. Ứng dụng đầu tiên bọn mình chọn đó là nhận diện biển số xe. Quá trình nhận diện bảng số xe bao gồm các bước sau đây
### Phát hiện bảng số xe trong hình
### Đọc chữ trên biển số vừa tìm được
Trong phần 1 này mình sẽ hướng dẫn các bạn phát hiện bảng số xe đơn giản bằng cách sử dụng công cụ openCV. Phương pháp này sử dụng các phép toán xử lý ảnh thông thường nên sẽ chạy rất nhanh so với các phương pháp Deep Learning.

### 2. Phát hiện bảng số xe <a name="notation"></a>
Cài đặt môi trường làm viêc: Để bắt đầu làm việc bạn tạo môi trường ảo có cài đặt python và opencv như sau:
```onda create -n NumberPlateRecognition python=2.7 opencv=2.4```
Các bước để phát hiện bảng số xe bao gồm:
Bước 1: Phát hiện cạnh trong hình băn
```
# Đọc file hình ảnh lên
image = cv2.imread('images.jpeg')

# Thay đổi kích thước hình ảnh. Chiều rộng là 500 và giữ nguyên tỉ lệ hình ảnh.
image = imutils.resize(image, width=500)

# Hiển thị hình ảnh ban đầu
cv2.imshow("Original Image", image)

# Chuyển ảnh màu qua ảnh trắng đen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

#
Loại bỏ nhiễu bằng phương pháp iterative bilateral filter. Phép biến đổi này giúp loại các nhiễu và giữ lại các cạnh trong hình ảnh
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

# Tìm các Edges trong hình trắng đen
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

# Tìm các countours (đường viền) trong hình ảnh
( cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = []
# Với mỗi đường viền, ta sẽ xấp xỉ bằng một hình đa giác lồi. Nếu đa giác có 4 cạnh thì đó có khả năng là bảng số
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Chọn các đường viền có đa giác xấp xỉ là 4 đỉnh
            NumberPlateCnt.append(approx) # Thêm da giác vào tập bảng số


# Vẽ các bảng số
for plate in NumberPlateCnt
    cv2.drawContours(image, [plate], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed

```


### 4. Kết bài <a name="conclusion"></a>





