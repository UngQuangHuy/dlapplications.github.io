---
layout: post
title: Nhận diện bảng số xe (Phần 1)
subtitle: Deep learning ứng dụng
hidden: false
tags: [Deep learning ứng dụng]
math: true
---

# Nhận diện bảng số xe (Phần 1)
### Mục lục:
1. [Mở đầu](#intro)
2. [Phát hiện bảng số xe](#method )
3. [Hạn chế của phương pháp ](#limitation)
4. [Kết bài](#conclusion)


### 1. Mở đầu <a name="intro"></a>
Hôm nay nhóm sẽ bắt đầu một số bài viết giúp bạn thực hiện một ứng dụng thực tế. Ứng dụng đầu tiên bọn mình chọn đó là nhận diện biển số xe. Quá trình nhận diện bảng số xe bao gồm các bước sau đây  
### Phát hiện bảng số xe trong hình  
### Đọc chữ trên biển số vừa tìm được  
Trong phần 1 này mình sẽ hướng dẫn các bạn phát hiện bảng số xe đơn giản bằng cách sử dụng công cụ openCV. Phương pháp này sử dụng các phép toán xử lý ảnh thông thường nên sẽ chạy rất nhanh so với các phương pháp Deep Learning.  

### 2. Phát hiện bảng số xe <a name="method"></a>
Cài đặt môi trường làm viêc: Để bắt đầu làm việc bạn tạo môi trường ảo có cài đặt python và opencv như sau:  
```conda create -n NumberPlateRecognition python=2.7 opencv=2.4```
Cài đặt thư viện imutils  
```pip install imutils```  

Các bước để phát hiện bảng số xe bao gồm:  
+ Đọc ảnh lên.  

![ảnh gốc](/img/20180706/images.jpeg)

+ Chuyển ảnh màu qua ảnh xám.  

![ảnh xám](/img/20180706/image_gray.jpeg)
+ Loại bỏ nhiễu bằng phương pháp iterative bilateral filter. Phép biến đổi này giúp loại các nhiễu và giữ lại các cạnh trong hình ảnh.  


![ảnh đã loại bỏ nhiễu](/img/20180706/image_bilateral.jpeg)
+ Tìm các Edges trong hình xám. Ở đây mình dung phương pháp canny.  

![ảnh sau khi tìm các edges](/img/20180706/image_cany.jpeg)
+ Tìm các countours (đường viền) trong hình ảnh. Sau đó sắp xếp lại theo diện tích của các đường viền và loại bỏ các đường viền có diện tích nhỏ hơn 30.   

![ảnh chứa các đường viền có diện tích lớn hơn 30](/img/20180706/image_countour.jpeg)
+ Với mỗi đường viền, ta sẽ xấp xỉ bằng một hình đa giác lồi. Nếu đa giác có 4 cạnh thì đó có khả năng là bảng số.  

![kết quả](/img/20180706/image_result.jpeg)

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
### 3.Ưu nhược điểm của phương pháp <a name="limitation"></a>)
### Ưu điểm
+ Phương pháp đơn giản, nhanh, nhỏ gọn. Có thể chạy trên rất nhiều platform như windows, linux, mac, ...  
+ Không cần dữ liệu để huấn luyện.  
Một số ví dụ phát hiện được bảng số

![ảnh xám](/img/20180706/OK1.jpeg)   

![ảnh xám](/img/20180706/OK2.jpeg)  


### Nhược điểm
Phương pháp mình giới thiệu trong bài này khá đơn giản nên nó sẽ có một số nhược điểm nhứ sau:
+ khi kích thước biển số xe trong hình thay đổi (ví dụ qua lớn hoặc quá bé) thì phương pháp sẽ gặp lỗi. Nên khi sử dụng chúng ta phải có ràng buộc khoảng cách từ camera và xe là cố định thì phương pháp sẽ chạy khá ổn.  
+ Phương pháp này sẽ gặp vấn đề nếu ảnh bị nhoè, độ phân giải thấp hoặc điều kiện ánh sáng không tốt. Nhưng điều này các bạn có thể kiểm chứng bằng thực tế nhé.  

Sau đây là một số ví dụ chưa phát hiện được bảng số


![ảnh xám](/img/20180706/Error1.jpeg)   

![ảnh xám](/img/20180706/Error2.jpeg)  


### 4. Kết bài <a name="conclusion"></a>
Trong bài này mình giới thiệu cho các bạn phương pháp phát hiện bảng số đơn giản bằng xử lý hình ảnh. Trong bài tiếp theo mình sẽ tiếp tục bước tiếp theo là nhận diện số và chữ trong bảng số.  




