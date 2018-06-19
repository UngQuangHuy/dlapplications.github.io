---
layout: post
title: Cài đặt tensorflow trong môi trường ảo Anaconda để làm deep learning.
hidden: true
tags: [cài đặt, tensorflow, môi trường ảo anaconda]
math: true
---
### 1. Giới thiệu <a name="intro"></a>
Như đã nói nhiều ở các post trước, muốn làm deep learning thì nên sử dụng tensorflow vì framework này được phát triển bởi google nên bạn sẽ được hỗ trợ rất nhiều từ google cũng như cộng đồng sử dụng tensorflow. Bọn mình tính làm tutorial hướng dẫn cụ thể cách cài Tensorflow trên Anaconda, tuy nhiên google 1 cái thì có khá nhiều clip hướng dẫn, nên ở đây mình sẽ sử dụng lại một clip hướng dẫn trong Deep learning course (cs-hcmup-2016) và giải thích rõ vì sao nên xài tensorflow trên môi trường ảo và giải thích các lệnh cơ bản trên Anacoda để bạn có thể làm việc trong hệ điều hành linux.  

<iframe width="560" height="315" src="https://www.youtube.com/watch?v=t_pxnHpRszg" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>  


### 2. Vì sao chúng ta nên sử dụng Anaconda <a name="notation"></a>

Tensorflow ra đời vào tháng 11 năm 2015, tính đến này được khoảng 2 năm rưỡi.
Version hiện tại là 1.8. Mình bắt đầu sử dụng Tensorflow khoảng hơn 1 năm trước, lúc đó nó đang ở version 1.0. Như thế trong hơn 1 năm Tensorflow dã update version 8 lần. Do tính chất là một framework opensource, có sự đóng góp rất lớn từ cộng đồng chứ không chỉ riêng các kỹ sư từ google, nên các API, hàm và cách sử dụng được update liên tục. Nhiều lần update các API và cách gọi đã bị thay đổi rất nhiều. Vì vậy, các bạn sẽ thấy mỗi source code bạn tìm được trên github thường mô tả cho các bạn version tensorflow mà họ sử dụng. Khi đó, bạn phải cài đúng version tensorflow thì mới có thể chạy được source code đó.  
Tuy nhiên khi cài Tensorflow trực tiếp vào máy tính, bạn chỉ có thể cài 1 version. Như thế bạn không thể làm việc với 2 project có version tensorflow khác nhau. Để giải quyết vấn đề này, Anaconda sẽ tạo ra các môi trường ảo khác nhau. Bạn có thể xem mỗi môi trường ảo là một máy tính, bạn có thể cài đặt các phần mềm bạn muốn sử dụng. Khi bạn vào một môi trường ảo, bạn có thể sử dụng các phần mềm đã cài trong môi trường ảo hiện tại. Như thế bạn có thể cài 1 version của tensorflow vào 1 môi trường ảo để làm việc với 1 hoặc nhiều project sử dụng cùng version tensorflow.


### 3. Tổng hợp các lệnh cơ bản của anaconda <a name="forward"></a>
Các bạn có thể thực hành theo video đính kèm. Mình xin tổng tổng hợp 1 số lệnh cơ bản của anaconda:
+ Tạo 1 môi trường mới  
'conda create --name [tên môi trường ảo]'  
ví dụ: 'conda create --name tensorflow1.8'  
+ Tạo 1 môi trường mới có cài đặt sẳn 1 phần mềm  
'conda create -n myenv [tên software]=[version]'  
Ví dụ: tạo môi trường tên tensorflow1.8 có cài đặt sẳn python 2.7  
'conda create -n tensorflow1.8 python=2.7'  

Một chú ý là bạn nên đặt tên các môi trường ảo ứng với version tensflow mà bạn cài đặt để tránh nhầm lẫn. Ví dụ như: tensorflow1.0, tensorflow1.8.  

+ đi vào 1 môi trường ảo  
'source activate [tên môi trường ảo]'  
+ Thoát khỏi 1 môi trường ảo  
'source deactivate'  

+ liệt kê tất cả môi trường ảo trong máy đã tạo  
'conda env list'  
+ xoá một môi trường ảo  
'conda remove --name [tên môi trường ảo] --all'  
+ cài đặt các gói phần mềm  
Sau khi vào một môi trường ảo, bạn có thể cài đặt phần mềm mới như sau: vào trang Anaconda.org và tìm theo tên phần mềm cần cài đặt. Ví dụ bạn gõ opencv. Bàn sẽ nhìn thấy nhiều version khác nhau. click vào version muốn cài đặt, bạn sẽ thấy câu lệnh bắt đầu bằng conda install .ở đây mình chọn opencv 3.4.1 thì bạn sẽ thầy câu lệnh là conda install -c conda-forge opencv. Copy câu này lên terminal chạy. Nó sẽ tự động cài phần mềm này.  
Ngoài ra, bạn cũng có thể cài đặt thông qua lệnh pip, nhưng nhớ cài đặt pip trước khi sử dụng nó nhé. pip install [tên phần mềm]  
+ Kiểm tra các phần mềm đã cài đặt  
'conda list'




Lê Đức Anh






