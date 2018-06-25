---
layout: post
title: Cài đặt tensorflow/pytorch/jupyter notebook trong môi trường ảo Anaconda để làm deep learning.
hidden: true
subtitle: Deep learning cơ bản
tags: [cài đặt, tensorflow, pytorch, jupyter notebook, môi trường ảo anaconda]
math: true
---

TODO: để tags một chữ cái thôi nhé. Sau này khó search

TODO: bài này nếu trong series  Deep Learning và Ứng Dụng  thì thêm tag dl_ap 

# Cài đặt tensorflow/pytorch/jupyter notebook trong môi trường ảo Anaconda để làm deep learning.
### Mục lục:
1. [Giới thiệu](#intro)
2. [Cài đặt cuda và cudnn cho GPU ](#gpu )
3. [Cài đặt Anaconda](#anaconda)
4. [ Cài đặt framework để làm deep learning](#framework)

4. [Kết bài](#conclusion)


### 1. Giới thiệu <a name="intro"></a>
Như đã nói nhiều ở [Làm sao để trở thành AI/Deep Learning engineer](https://www.facebook.com/permalink.php?story_fbid=212067389390163&id=204704190126483), [Sơ lược về các Deep Learning Framework](https://dlapplications.github.io/2018-06-13-dlframework/), muốn phát triển các ứng dụng sử dụng deep learning một cách dễ dàng thì nên sử dụng framework phù hợp. Như đánh gia trong bài [Sơ lược về các Deep Learning Framework](https://dlapplications.github.io/2018-06-13-dlframework/) thì tensorflow, pytorch là 2 framework đơn giản và dễ sử dụng nhất. Đặc biệt là cả hai đều có cộng đồng người dùng và model zoo rất lớn. Như vậy, khi gặp một vấn đề khó giải quyết, bạn sẽ được hỗ trợ rất nhiều từ cộng đồng sử dụng khi gặp vấn đề hoặc muốn đặt câu hỏi. Cộng đồng [Tensorflow](https://www.tensorflow.org/community/),  [Pytorch](https://discuss.pytorch.org/).  

Trong post này nhóm xin giới thiệu cách cài đặt 2 framework tensorflow và pytorch. Đây là 2 framework chính được sử dụng trong những bài viết tới. 

Trong bài này mình hướng dẫn cài đặt các framework lên máy tính cấu hình như sau Ubuntu 16.04, 64 bit, GPU GTX 1070. Nếu bạn không có card GPU có thể cài đặt framework phiên bản CPU.


### 2. Cài đặt cuda và cudnn cho GPU <a name="gpu"></a>
# Cài đặt cuda
Bạn vào đường link của [Nvidia](https://developer.nvidia.com/cuda-90-download-archive) và chọn các tab như hình phía dưới để download được cuda 9.0.

![download cuda 9.0 và hướng dẫn cài đặt](/img/20180618/img_1.png)  
Chạy 4 câu lệnh sau để cài đặt cuda
```sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb```
```sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub```
```sudo apt-get update```
```sudo apt-get install cuda```


# Cài đặt cudnn
Bạn vào đường [link này](https://developer.nvidia.com/cudnn) để download cudnn 7.1. Bạn cần phải đăng kí một tài khoản để có thể download cudnn nhé.  
Unzip file vừa download vào folder Downloads. Bạn sẽ thấy folder cuda có chứa 2 folder con là include và lib64. Copy 2 folder này vào thư mục CUDA_HOME bằng các câu lệnh sau:

```sudo cp cuda/include/cudnn.h /usr/local/cuda/include```
```sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64```
```sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*```

### 3. Cài đặt Anaconda <a name="anaconda"></a>
# Vì sao nên sử dụng môi trường ảo
Tensorflow ra đời vào tháng 11 năm 2015, tính đến này được khoảng 2 năm rưỡi.
Version hiện tại là 1.8. Mình bắt đầu sử dụng Tensorflow khoảng hơn 1 năm trước, lúc đó nó đang ở version 1.0. Như thế trong hơn 1 năm Tensorflow dã update version 8 lần. Tương tự, pytorch cũng có rất nhiều version khác nhau. Do tính chất là framework opensource, có sự đóng góp rất lớn từ cộng đồng, nên các API, hàm và cách sử dụng được update liên tục. Nhiều lần update các API và cách gọi đã bị thay đổi rất nhiều. Vì vậy, các bạn sẽ thấy mỗi source code bạn tìm được trên github thường mô tả cho các bạn version tensorflow/pytorch mà họ sử dụng. Khi đó, bạn phải cài đúng version tensorflow/pytorch thì mới có thể chạy được source code đó.  
Tuy nhiên khi cài Tensorflow/pytorch trực tiếp vào máy tính, bạn chỉ có thể cài 1 version. Như thế bạn không thể làm việc với 2 project có version tensorflow/pytorch khác nhau. Để giải quyết vấn đề này, Anaconda sẽ tạo ra các môi trường ảo khác nhau. Bạn có thể xem mỗi môi trường ảo là một máy tính, bạn có thể cài đặt các phần mềm bạn muốn sử dụng. Khi bạn vào một môi trường ảo, bạn có thể sử dụng các phần mềm đã cài trong môi trường ảo hiện tại. Như thế bạn có thể cài 1 version của tensorflow/pytorch vào 1 môi trường ảo để làm việc với 1 hoặc nhiều project sử dụng cùng version tensorflow/pytorch.

# Cài đặt Anaconda  
Bạn vào đường [link này](https://www.anaconda.com/download/#linux) để download Anaconda cho python 2.7.  
Bạn chạy câu lệnh sau trên terminal để cài đặt  
```bash Anaconda-latest-Linux-x86_64.sh```


### 4. Cài đặt framework để làm deep learning <a name="framwork"></a>

# Cài đặt tensorflow  

Tạo môi trường tên tensorflow1.8 có cài đặt sẳn python 2.7  
```conda create -n tensorflowGPU1.8 python=2.7```  
 Đi vào 1 môi trường ảo  
 ```source activate tensorflow1.8```  
 Cài đặt tensorflow  
 + Dành cho máy có gắn GPU.
  ```conda install -c aaronzs tensorflow-gpu ```  
 + Dành cho máy không có GPU  
   ```conda install -c conda-forge tensorflow ```  
Nếu bạn muốn cài đặt phiên bản khác, bạn có thể vào trang Anaconda.org để tìm.  
 
 
# Cài đặt Pytorch  
Tạo môi trường tên pytorch có cài đặt sẳn python 2.7  
```conda create -n pytorch python=2.7```  
 Đi vào 1 môi trường ảo  
 ```source activate pytorch```  
 Cài đặt pytorch  
 + Dành cho máy có gắn GPU.
  ```conda install pytorch torchvision cuda90 -c pytorch  ```  
 + Dành cho máy không có GPU.  
   ```conda install pytorch-cpu torchvision-cpu -c pytorch```  
Câu lệnh trên giúp bạn cài đặt pytorch phiên bản mới nhất. Nếu bạn muốn cài đặt các version khác thì tham khảo hướng dẫn ở  [link này](https://pytorch.org/previous-versions/).

# Kiểm tra cài đặt  
Sau khi cài đặt xong, gõ lệnh python để vào màn hình chạy python. Chạy các lệnh sau để xác nhận cài đặt thành công
+ Tensorflow  
 ```import tensorflow as tf ```
 ```hello = tf.constant('Hello, TensorFlow!') ```
 ```sess = tf.Session() ```
 ```print(sess.run(hello)) ```
Nếu bạn thấy dòng chữ ''Hello, TensorFlow!'' trên màn hình thì quá trình cài đặt tensorflow đã thành công.  
+ pytorch  
 ```import torch ```
 ```x = torch.randn(2, 4) ```
 ```x  ```
Nếu bạn thấy 1 ma trận 2 hàng 4 cột và dòng chữ 'torch.FloatTensor of size 2x4'' trên màn hình thì quá trình cài đặt pytorch đã thành công.  


# Cài đặt jupyter notebook 
Jupyter notebook là phần mềm giúp bạn tạo ra các tài liệu vừa chứa text, công thức, hình ảnh và cả code có thể thực thi. Đây là một công cụ rất tốt để tạo ra các bài tutorial dễ hiểu và trực quan. Nhóm sẽ sử dụng Jupyter notebook để tạo các bài học về deep learning cho các bạn. 
```conda install jupyter```  

+ Chạy jupyter notebook  
```jupyter notebook```



# Tổng hợp các lệnh cơ bản của Anaconda 
+ Tạo 1 môi trường mới  
```conda create --name [tên môi trường ảo]```
ví dụ: ```conda create --name tensorflow1.8``` 
+ Tạo 1 môi trường mới có cài đặt sẳn 1 phần mềm  
```conda create -n myenv [tên software]=[version]``` 
Ví dụ: tạo môi trường tên tensorflow1.8 có cài đặt sẳn python 2.7  
```conda create -n tensorflow1.8 python=2.7```  

Một chú ý là bạn nên đặt tên các môi trường ảo ứng với version tensflow mà bạn cài đặt để tránh nhầm lẫn. Ví dụ như: tensorflow1.0, tensorflow1.8, pytorch0.1.12.  

+ Đi vào 1 môi trường ảo  
```source activate [tên môi trường ảo]```  
+ Thoát khỏi 1 môi trường ảo  
```source deactivate```

+ Liệt kê tất cả môi trường ảo trong máy đã tạo  
```conda env list```  
+ Xoá một môi trường ảo  
```conda remove --name [tên môi trường ảo] --all```  
+ Cài đặt các gói phần mềm  
Sau khi vào một môi trường ảo, bạn có thể cài đặt phần mềm mới như sau: vào trang Anaconda.org và tìm theo tên phần mềm cần cài đặt. Ví dụ bạn gõ opencv. Bàn sẽ nhìn thấy nhiều version khác nhau. click vào version muốn cài đặt, bạn sẽ thấy câu lệnh bắt đầu bằng ```conda install``` . Ở đây mình chọn opencv 3.4.1 thì bạn sẽ thầy câu lệnh là ```conda install -c conda-forge opencv```. Copy câu này lên terminal chạy. Nó sẽ tự động cài phần mềm này.  

+ Kiểm tra các phần mềm đã cài đặt  
```conda list```



### 4. Kết bài <a name="conclusion"></a>
Vậy là trong phần này chúng ta đã cài đăt được các framework và các công cụ cần thiết để có thể bắt đầu học và làm deep learning. 
Nhóm sẽ cố gắng giúp các bạn học Deep Learning thông qua các bài viết về lí thuyết cũng như thực hành trong tương lai thông qua hiểu biết và kinh nghiệm của các thành viên. Mong các bạn tiếp tục ủng hộ nhóm



Lê Đức Anh






