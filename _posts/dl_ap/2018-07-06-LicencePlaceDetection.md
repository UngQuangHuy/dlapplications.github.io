---
layout: post
title: Nhận diện bảng số xe (Phần 1)
subtitle: Deep learning ứng dụng
hidden: false
tags: [Nhận diện bảng số , dl_ap]
math: true
---

# Nhận diện bảng số xe (Phần 1)
### Mục lục:
1. [Mở đầu](#intro)
2. [Cài đặt](#install )
3. [Phát hiện bảng số xe](#method )
4. [Ưu nhược điểm của phương pháp ](#limitation)
5. [Kết bài](#conclusion)


### 1. Mở đầu <a name="intro"></a>
Hôm nay nhóm sẽ bắt đầu một số bài viết giúp bạn thực hiện một ứng dụng thực tế. Ứng dụng đầu tiên bọn mình chọn đó là nhận diện biển số xe. Quá trình nhận diện bảng số xe bao gồm các bước sau đây  
I. Phát hiện bảng số xe trong hình  
II. Đọc chữ trên biển số vừa tìm được  
Trong phần 1 này mình sẽ hướng dẫn các bạn phát hiện bảng số xe. Do chưa chuẩn bị được dữ liệu và máy để huấn luyện mô hình 
Deep Learning, trong bài này mình hướng dẫn phương pháp đơn giản bằng cách sử dụng công cụ openCV. Phương pháp này sử dụng các phép toán xử lý ảnh thông thường nên không cần dữ liệu để huấn luyện và chạy rất nhanh so với các phương pháp Deep Learning.  

### 2. Cài đặt <a name="install"></a>
Cài đặt môi trường làm viêc: Để bắt đầu làm việc bạn tạo môi trường ảo có cài đặt python và opencv như sau:  
```conda create -n NumberPlateRecognition python=2.7 opencv=2.4```
Cài đặt thư viện imutils  
```pip install imutils```  
### 3. Phát hiện bảng số xe <a name="method"></a>
Các bước để phát hiện bảng số xe bao gồm:  
+ Đọc ảnh lên và thay đổi kích thước.  
```
image = cv2.imread('images.jpeg')
image = imutils.resize(image, width=500)
```  

![ảnh gốc](/img/20180706/images.jpeg)

+ Chuyển ảnh màu qua ảnh xám.  
```gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)```  

![ảnh xám](/img/20180706/image_gray.jpeg)  

+ Loại bỏ nhiễu bằng phương pháp làm mịn. Có khá nhiều phương pháp làm mịn như lấy trung bình của các pixel lân cận, hay dùng phân bố Gaussian. Tuy nhiên các phương pháp này loại bỏ được nhiễu nhưng cũng làm mất đi các cạnh (Edges) trong hình. Các cạnh này là thông tin quan trọng để chúng ta xác định vùng bảng số nên cần phải giữ lại. Phương pháp iterative bilateral filter giúp chúng ta vừa giúp làm mịn lại tránh loại bỏ cạnh trong hình ảnh. Các bạn có thể tham khảo [chi tiết tại đây](http://eric-yuan.me/bilateral-filtering/)  
```gray = cv2.bilateralFilter(gray, 11, 17, 17)```

![ảnh đã loại bỏ nhiễu](/img/20180706/image_bilateral.jpeg)  

+ Tìm các cạnh trong hình đã được làm mịn ở trên. Ở đây mình dùng phương pháp [Canny edge detection](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html).  
```edged = cv2.Canny(gray, 170, 200)```  

![ảnh sau khi tìm các edges](/img/20180706/image_cany.jpeg)  

+ Tìm các countours (các vùng kết nối với nhau) trong hình ảnh. Sau đó sắp xếp lại theo diện tích của các đường viền và loại bỏ các đường viền có diện tích nhỏ hơn 30.   

```
( cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
```  

![ảnh chứa các đường viền có diện tích lớn hơn 30](/img/20180706/image_countour.jpeg)  

+ Với mỗi đường viền, ta sẽ xấp xỉ bằng một hình đa giác lồi. Nếu đa giác có 4 cạnh thì đó có khả năng là bảng số.  
```
for c in cnts:
   peri = cv2.arcLength(c, True)
   approx = cv2.approxPolyDP(c, 0.02 * peri, True)
   cv2.drawContours(tmp, [c], -1, (0,0,128), 2)
   if len(approx) == 4:
       NumberPlateCnt.append(approx)
```  

+ Vẽ lại các bảng số đã tìm được.  
```
for plate in NumberPlateCnt:
    cv2.drawContours(image, [plate], -1, (0,255,0), 2)
```  

![kết quả](/img/20180706/image_result.jpeg)  

Các bạn có thể sử dụng code tổng hợp [tại đây](https://github.com/dlapplications/License-Plate-Recognition/blob/master/plateDetection.py)


### 4.Ưu nhược điểm của phương pháp <a name="limitation"></a>)
### Ưu điểm
+ Phương pháp đơn giản, nhanh, nhỏ gọn. Có thể chạy trên rất nhiều platform như windows, linux, mac, ...  
+ Không cần dữ liệu để huấn luyện.  
Một số ví dụ phát hiện được bảng số

![ảnh nhận diện đúng 1](/img/20180706/OK1.jpeg)   

![ảnh  nhận diện đúng 2](/img/20180706/Ok2.jpeg)  

Đối với các vung phát hiện sai ở trên, chúng ta có thể kết hợp với quá trình đọc chữ để loại bỏ các vùng không phải bảng số.


### Nhược điểm
Phương pháp mình giới thiệu trong bài này khá đơn giản nên nó sẽ có một số nhược điểm như sau:
+ khi kích thước biển số xe trong hình thay đổi (ví dụ qua lớn hoặc quá bé) thì phương pháp sẽ gặp lỗi. Nên khi sử dụng chúng ta phải có ràng buộc khoảng cách từ camera và xe là cố định. Khi đó kích thước biển số không thay đổi nhiều thì phương pháp sẽ chạy khá ổn.  
+ Phương pháp này sẽ gặp vấn đề nếu ảnh bị nhoè, độ phân giải thấp hoặc điều kiện ánh sáng không tốt. Nhưng điều này các bạn có thể kiểm chứng bằng thực tế nhé.  

Sau đây là một số ví dụ chưa phát hiện được bảng số


![ảnh nhận diện sai 1](/img/20180706/Error1.jpeg)   

![ảnh nhận diện sai 1](/img/20180706/Error2.jpeg)  

Nếu các bạn muốn khắc phục các nhược điểm trên, chúng ta có thể sử dụng object detection trong deep learning. Mình sẽ cố gắng gửi đến các bạn trong thời gian sớm nhất.

### 5. Kết bài <a name="conclusion"></a>
Trong bài này mình giới thiệu cho các bạn phương pháp phát hiện bảng số đơn giản bằng xử lý hình ảnh. Trong bài tiếp theo mình sẽ tiếp tục bước tiếp theo là nhận diện số và chữ trong bảng số.  




