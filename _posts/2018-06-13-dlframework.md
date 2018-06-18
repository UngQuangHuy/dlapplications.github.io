---
layout: post
title: Deep Learning Framework
subtitle: Deep Learning và Ứng Dụng Phần 2.
tags: [tutorial, begin, dl_ap]
comment: true
hidden: true
---

# Sơ lược về các Deep Learning Framework

## Mục lục

### Mục lục:
1. [Mở đầu](#intro)
2. [Lựa chọn như thế nào](#tools)

    2.1 [Các Framework thường dùng](#tools_1)

    2.2 [Các tiêu chí chọn lựa Framework](#tools_2)

4. [Kết luận](#end)


## Mở đầu <a name="intro"></a>

Ở bài trước [ml-quick-guide](https://dlapplications.github.io/2018-06-02-ml-quick-guide/) 
chúng ta đã biết ngành Deep Learning có qui trình phát triển gồm 6 bước như sau. 

1. Định nghĩa vấn đề (Problem definition)
2. Thu thập dữ liệu (Data Gathering)
3. Lọc dữ liệu (Data Parsing)
4. Training (tạo mô hình)
5. Testing (kiểm tra độ chính xác)
6. Deploying (triển khai trên sản phẩm)

Deep Learning Framework là một phần mềm (software) mà chúng ta dùng để thực hiện bước 4, 5 và 6. Quá trình phát triển của Deep Learning phải qua nhiều lần tinh chỉnh để đạt Accuracy cao nhất. Lựa chọn một framework phù hợp giúp chúng ta dễ dàng phát triển hơn, đẩy nhanh quá trình ra sản phẩm. 

## Lựa chọn như thế nào <a name="tools"></a>

Hiện tại như liệt kê ở  [Deep Learning framework](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software) có tới hàng chục Deep Learning Framework. Điều này tạo khó khăn rất lớn cho người mới bắt đầu (newbie). Mỗi framework đều có một công ty đứng sau, đều tốt, và đều dùng được trong Training, Testing và Deploying (bước 4, 5, 6 kể trên). Không có framework nào là tốt nhất, chúng ta chỉ có thể dựa trên các tiêu chí của dự án (project) để chọn ra framework thích hợp mà thôi.

### Các Deep Learning framework thường dùng  <a name="tools_1"></a>

Danh sách 5 framework thường dùng. 

| Tên        | Quy mô của model zoo   | Interface  | OpenMP support |
| ---------- |:-------------:| -----:     | -------:|
| Caffe       | nhiều  | C++, Matlab, Python      | Yes |
| MXNet      | Nhiều      |    C++, Python, Julia, Matlab, JavaScript, Go, R, Scala, Perl | Yes |
| Tensorflow | Rất nhiều      |    Python, C++, Java, Keras | No |
| Chainer    | Nhiều      |   Python    | No |
| Pytorch    | Nhiều      |    Python | Yes |

Chú ý: mỗi framework có một format riêng. Training, Testing, Deploying phải bằng một framework duy nhất. Trong trường hợp bất khả kháng, phải dùng Framework khác để Deploying chúng ta có thể sử dụng  [onnx](https://github.com/onnx) để chuyển đổi giữa các format với nhau. 

### Các tiêu chí chọn lựa Deep Learning Framework <a name="tools_2"></a>

Khi chọn lưa Deep Learning Framework, chúng ta thường dùng những tiêu chí sau đây.

1. Có model zoo, công khai nhiều model có thể tái sử dụng
2. Hỗ trợ việc load, chạy model bằng C 
3. Dễ viết, dễ sử dụng

Ở phần tiếp theo, nhóm sẽ đánh giá 5 Framework này theo 3 tiêu chí kể trên. 

#### Quy mô của model zoo <a name="tools_3"></a>

Sau khi định nghĩa vấn đề (Problem Definition) xong, problem sẽ được chia thành các task nhỏ hơn. Mỗi task sẽ được giải quyết bằng một lớp Deep Learning. Việc training từ con số 0 chỗ mỗi model tương ứng với các lớp (hay task) rất mất thời gian, và cũng không đảm bảo sẽ đạt Accuracy mong muốn. 

Vì vậy, bước Training thường được bắt đầu bằng cách tìm những model có thể tái sử dụng cho mỗi task. Sau đấy chúng ta sẽ training lại (finetune) bằng dataset khác để chỉnh model về trạng thái mong muốn. Cho nên việc có model zoo lớn  và công khai nhiều model để chúng ta tái sử dụng là tiêu chí quan trọng nhất. 

Cả 5 framework ở trên đều có model zoo rất lớn và phong phú, chứa hầu như toàn bộ các nghiên cứu (research) nổi tiếng. 

1. [Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
2. [Mxnet model zoo](https://mxnet.incubator.apache.org/model_zoo/index.html)
3. [Tensorflow](https://github.com/tensorflow/models)
5. [Chainer model zoo](https://github.com/chainer/chainercv)
4. [Pytorch model zoo](https://github.com/pytorch/vision)



| Tên        | Quy mô của model zoo   | 
| ---------- |:-------------:| 
| Caffe       | 4  | 
| MXNet      | 4      |  
| Tensorflow | 5      |
| Chainer    | 4      |  
| Pytorch    | 4      |   


#### Hỗ trợ việc load, sử dụng model bằng C <a name="tools_4"></a>

Sau khi training xong và có được model với Accuracy mong muốn, model này sẽ được Deploying lên trên thiết bị đầu cuối. Ví dụ như: smartphone, embedded system, mạch FPGA,  etc 

Một Deep Learning Framework bao gồm rất nhiều dependency, library, tools khác nhau. Số lượng thiết bị đầu cuối (board, device) là rất lớn với đủ mọi architecture, chipset khác nhau. Điểm sơ trong [elinux](https://elinux.org/Main_Page) chúng ta có thấy tầm 20 loại board khác nhau, chưa kể các board ít người dùng không được liệt kê khác. Việc đảm bảo một Framework chạy được trên tất cả các board, device là bất khả thi.

Để đảm bảo model tạo ra có thể chạy trên mọi loại board, các Deep Learning Framework hiện nay có một chức năng rất đặc biệt. Đó là xuất ra (export) chỉ các library cần thiết cho việc load và sử dụng model. Các library này sẽ nằm trong một file header C duy nhất. Chỉ việc đem file C này và compile trên board tương ứng là có thể load và sử dụng model được. 
 
Khi sử dụng model trên các board đầu cuối, quá trình tính toán phải được tăng tốc bằng cách ép toàn bộ core tham gia vào quá trình tính toán (Utilize all the cores). Phương thức đơn giản nhất để làm việc này là sử dụng  OpenMP.

![Scene_text](/img/20180613/openMP.png) Performance Optimization on Modern Platforms

Chú ý: board FPGA là trường hợp đặc biệt, nên trực tiếp sử dụng library và tools của Xilinx để đạt hiệu quả cao nhất. 

Tensorflow và MXNet là tốt nhất trong tiêu chí này. Pytorch hiện cũng có kế hoạch tương tự [Pytorch 1.0](https://pytorch.org/2018/05/02/road-to-1.0.html), nhưng chúng ta phải chờ đến cuối năm 2018 mới bắt đầu được sử dụng . Caffe tuy không có chức năng này nhưng vì được viết bằng C nên vẫn có thể compile được. Chainer thì vừa không có chức năng này, vừa không thể compile được.


| Tên        | Hỗ trợ việc load, chạy model bằng C    | 
| ---------- |:-------------:| 
| Caffe       | 2  | 
| MXNet      | 5      |  
| Tensorflow | 5      |
| Chainer    | 1      |  
| Pytorch    | 3      |   

#### Dễ viết, dễ sử dụng <a name="tools_5"></a>

Đây là tiêu chí khó đánh giá nhất vì quen dùng cái gì thì nó thành dễ nhất thôi. Theo cảm nhận chủ quan thì Chainer và Pytorch là 2 framework dễ viết và dễ sử dụng nhất. Với 2 framework này chỉ cần một tuần là có thể thông thạo được cách dùng. Trong khi với 3 framework còn lại có thể mất cả tháng. 

| Tên        | Dễ viết, dễ sử dụng   | 
| ---------- |:-------------:| 
| Caffe       | 2  | 
| MXNet      | 3      |  
| Tensorflow | 3      |
| Chainer    | 5      |  
| Pytorch    | 5      |   

## Kết luận <a name="end"></a>

Trong bài viết này chúng ta đã lướt sơ bộ qua 5 framework thường sử dụng trong Deep Learning. Không có framework hoàn hảo, tất cả đều mang trong mình điểm mạnh và điểm yếu riêng. Tùy vào mục đích sử dụng mà chúng ta lựa chọn framework thích hợp

Khi bắt đầu một dự án sử dụng Deep Learning, nếu chúng ta chưa biết hoặc không xác định được thiết bị cuối là gì thì lời khuyên là nên sử dụng Mxnet hoặc Tensorflow. Tuy nhiên, nếu chúng ta chỉ muốn học Deep Learning thôi thì lời khuyên là sử dụng các Framework dễ viết và dễ sử dụng nhất. 

Đối tượng của series này là newbie, các bạn chưa có kiển thức chuyên sâu về Deep Learning. Để cho các bạn nhanh chóng nắm bắt được vấn đề, những bài còn lại của series sẽ sử dụng Pytorch làm framework chính. Một số trường hợp khi cần demo trên mobile hoặc board khác sẽ chuyển qua Mxnet hoặc Tensorflow.  

Vũ Gia Trường.

