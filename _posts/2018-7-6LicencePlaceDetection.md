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
```conda create -n NumberPlateRecognition python=2.7 opencv=2.4```
Cài đặt thư viện imutils  
```pip install imutils```  

Các bước để phát hiện bảng số xe bao gồm:  
+ Đọc ảnh lên.  

![ảnh gốc]()

+ Chuyển ảnh màu qua ảnh xám.  

![ảnh xám]()
+ Loại bỏ nhiễu bằng phương pháp iterative bilateral filter. Phép biến đổi này giúp loại các nhiễu và giữ lại các cạnh trong hình ảnh.  


![ảnh đã loại bỏ nhiễu]()
+ Tìm các Edges trong hình xám. Ở đây mình dung phương pháp canny.  

![ảnh sau khi tìm các edges]()
+ Tìm các countours (đường viền) trong hình ảnh. Sau đó sắp xếp lại theo diện tích của các đường viền và loại bỏ các đường viền có diện tích nhỏ hơn 30.   

![ảnh chứa các đường viền có diện tích lớn hơn 30]()
+ Với mỗi đường viền, ta sẽ xấp xỉ bằng một hình đa giác lồi. Nếu đa giác có 4 cạnh thì đó có khả năng là bảng số.  

![kết quả]()

```
import numpy as np
import cv2
import  imutils

# Read image
image = cv2.imread('images.jpeg')

# Resize image 
image = imutils.resize(image, width=500)

# Display image
cv2.imshow("Original Image", image)

# Convert to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)
cv2.imwrite("image_gray.jpeg", gray)

# Remove noise while preserving edges by iterative bilateral filter
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
cv2.imwrite("image_bilateral.jpeg", gray)

# Find Edges by Canny method
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)
cv2.imwrite("image_cany.jpeg", edged)

# Find contours
( cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Sort contours based on their area and keep minimum required area as '30'
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = []
tmp = image.copy()
count = 0
# For each contour, approximate it by a polygon and select quadrilateral
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.drawContours(tmp, [c], -1, (0,0,128), 2)
    #cv2.drawContours(tmp, [approx], -1, (0,255,0), 2)
    if len(approx) == 4:
        NumberPlateCnt.append(approx)

cv2.imwrite("image_countour.jpeg", tmp)
# draw results
for plate in NumberPlateCnt:
    cv2.drawContours(image, [plate], -1, (0,255,0), 2)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.imwrite("image_result.jpeg", image)
cv2.waitKey(0)
```

### 4. Kết bài <a name="conclusion"></a>





