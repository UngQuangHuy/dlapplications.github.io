---
layout: post
title: Cài đặt tensorflow/pytorch/jupyter notebook trong môi trường ảo Anaconda để làm deep learning.
hidden: true
subtitle: Deep learning cơ bản
tags: [cài đặt, tensorflow, pytorch, jupyter notebook, môi trường ảo anaconda]
math: true
---
# Cài đặt tensorflow/pytorch/jupyter notebook trong môi trường ảo Anaconda để làm deep learning.
Deep learning cơ bản
### Mục lục:
1. [Giới thiệu](#intro)
2. [Vì sao chúng ta nên sử dụng Anaconda](#notation )
3. [Tổng hợp các lệnh cơ bản của anaconda](#forward)
4. [Kết bài](#conclusion)


### 1. Giới thiệu <a name="intro"></a>
Như đã nói nhiều ở các post trước, muốn làm deep learning thì nên sử dụng framework. Như đánh gia trong bài Sơ lược về các Deep Learning Framework thì tensorflow, pytorch là 2 framework dễ sử dụng, có cộng đồng sử dụng rất lớn. Như vậy bạn sẽ được hỗ trợ rất nhiều từ cộng đồng sử dụng khi gặp vấn đề hoặc muốn đặt câu hỏi. Trong post này mình xin giới thiệu cách cài đặt 2 framework này, và đây cũng là 2 framework bọn mình chọn để hướng dẫn các bạn làm deep learning trong tương lai.
Bạn có thể tham khảo clip hướng dẫn cụ thể cách cài Tensorflow trên Anaconda trong Deep learning course (cs-hcmup-2016). Ở đây mình sẽ giải thích rõ vì sao nên xài framework trên môi trường ảo, cơ bản trên Anacoda và làm sao để cài đặt tensorflow và pytorch trên Anacoda để bạn có thể làm việc trong hệ điều hành linux.  

Ngoài ra nhóm sẽ sử dụng Jupyter notebook để tạo các bài học về deep learning cho các bạn. Jupyter notebook là phần mềm giúp bạn tạo ra các tài liệu vừa chứa text, công thức, hình ảnh và cả code có thể thực thi. Đây là một công cụ rất tốt để tạo ra các bài tutorial dễ hiểu và trực quan. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/t_pxnHpRszg" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### 2. Vì sao chúng ta nên sử dụng Anaconda <a name="notation"></a>

Tensorflow ra đời vào tháng 11 năm 2015, tính đến này được khoảng 2 năm rưỡi.
Version hiện tại là 1.8. Mình bắt đầu sử dụng Tensorflow khoảng hơn 1 năm trước, lúc đó nó đang ở version 1.0. Như thế trong hơn 1 năm Tensorflow dã update version 8 lần. Tương tự, pytorch cũng có rất nhiều version khác nhau. Do tính chất là framework opensource, có sự đóng góp rất lớn từ cộng đồng, nên các API, hàm và cách sử dụng được update liên tục. Nhiều lần update các API và cách gọi đã bị thay đổi rất nhiều. Vì vậy, các bạn sẽ thấy mỗi source code bạn tìm được trên github thường mô tả cho các bạn version tensorflow/pytorch mà họ sử dụng. Khi đó, bạn phải cài đúng version tensorflow/pytorch thì mới có thể chạy được source code đó.  
Tuy nhiên khi cài Tensorflow/pytorch trực tiếp vào máy tính, bạn chỉ có thể cài 1 version. Như thế bạn không thể làm việc với 2 project có version tensorflow/pytorch khác nhau. Để giải quyết vấn đề này, Anaconda sẽ tạo ra các môi trường ảo khác nhau. Bạn có thể xem mỗi môi trường ảo là một máy tính, bạn có thể cài đặt các phần mềm bạn muốn sử dụng. Khi bạn vào một môi trường ảo, bạn có thể sử dụng các phần mềm đã cài trong môi trường ảo hiện tại. Như thế bạn có thể cài 1 version của tensorflow/pytorch vào 1 môi trường ảo để làm việc với 1 hoặc nhiều project sử dụng cùng version tensorflow/pytorch.


### 3. Tổng hợp các lệnh cơ bản của anaconda <a name="forward"></a>
Các bạn có thể thực hành theo video đính kèm. Mình xin tổng tổng hợp 1 số lệnh cơ bản của anaconda:
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
Ngoài ra, bạn cũng có thể cài đặt thông qua lệnh pip, nhưng nhớ cài đặt pip trước khi sử dụng nó nhé. pip install [tên phần mềm]  
+ Kiểm tra các phần mềm đã cài đặt  
```conda list```

+ Cài đặt tensorflow
```conda install -c conda-forge tensorflow ```
Câu lệnh trên giúp bạn cài đặt tensorflow phiên bản mới nhất. Nếu bạn muốn cài đặt các version khác thì tham khảo hướng dẫn ở [link này](https://www.tensorflow.org/install/install_linux#InstallingAnaconda).
+ Cài đặt pytorch
```conda install pytorch torchvision -c pytorch```
Câu lệnh trên giúp bạn cài đặt pytorch phiên bản mới nhất. Nếu bạn muốn cài đặt các version khác thì tham khảo hướng dẫn ở  [link này](https://pytorch.org/previous-versions/).

+ Cài đặt jupyter notebook  
```conda install jupyter```  

+ Chạy jupyter notebook  
```jupyter notebook```

### 4. Kết bài <a name="intro"></a>
Vậy là trong phần này chúng ta đã cài đăt được các framework và các công cụ cần thiết để có thể bắt đầu học và làm deep learning. Nhóm sẽ cố gắng giúp các bạn học Deep Learning thông qua các bài viết về lí thuyết cũng như thực hành trong tương lai thông qua hiểu biết và kinh nghiệm của các thành viên. Mong các bạn tiếp tục ủng hộ nhóm.

Lê Đức Anh






