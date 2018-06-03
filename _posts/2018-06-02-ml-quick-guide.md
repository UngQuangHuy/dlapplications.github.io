---
layout: post
title: Hướng dẫn sử dụng Deep Learning
subtitle: Quy trình chung cho phát triển sử dụng Deep Learning
tags: [tutorial, begin]
---


# Hướng dẫn sử dụng Deep Learning 

## Tóm tắt

AI (Artificial intelligent, trí tuệ nhân tạo) nói chung và Deep Learning nói riêng hiện nay đang là một trong những ngành nóng nhất trong bộ môn khoa học máy tính (Computer science). Chỉ trong vài năm ngắn ngủi, Deep Learning đã đẩy nhanh sự phát triển của các lĩnh vực nhỏ hơn như dịch, nhận diện giọng nói, nhận diện ảnh, ... Và đồng thời mang thành tựu của những lĩnh vực đấy từ trong phòng thí nghiệm ra với công chúng - điều mà vì nhiều lý do đã từng rất khó khăn.

Đồng hành với sự phát triển đó, đã có rất nhiều tutorial, khóa học, sách, video về  Deep Learning được phát hành. Chúng ta có thể dễ dàng tìm được những tài liệu với chất lượng rất cao nhưng lại hoàn toàn miễn phí. Tuy nhiên điều đó cũng dẫn đến hệ quả là với những người mới bắt đầu sẽ rất nhiều thời gian để chọn lọc được những thứ thật sự cần thiết với mình. 

Để giải quyết vấn đề đó, nhóm "Deep Learning và ứng dụng" đã bắt đầu series dài kì này. Series sẽ là tổng hợp quá trình phát triển những ứng dụng nhỏ sử dụng Deep Learning mà chúng ta có thể dễ dàng dùng lại được trong công việc cũng như cuộc sống hàng ngày. Ví dụ như: nhận diện chữ viết, đọc biển số xe, tìm nhãn hàng, tìm các barcode, vân vân. Mời các bạn cùng đón xem.

## Quy trình phát triển của Deep Learning

Deep Learning là một ngành nhỏ của Machine Learning. Vì thế cả hai có chung một quy trình phát triển như sau.

1. Định nghĩa vấn đề (Problem definition)
2. Thu thập dữ liệu (Data Gathering)
3. Lọc dữ liệu (Data Parsing)
4. Training (tạo mô hình)
5. Testing (kiểm tra độ chính xác)
6. Deploying (triển khai trên sản phẩm)

#### 1. Định nghĩa vấn đề (Problem Definition)

Đây là công đoạn quan trọng nhất, đòi hỏi nhiều kiến thức và kinh nghiệm. Với những problem đơn giản, chỉ cần một lớp Deep Learning (hay một model) là đủ để giải quyết. Tuy nhiên với những problem phức tạp cần nhiều lớp Deep Learning để giải quyết.

Ví dụ về một problem đơn giản: nhận diện khuôn mặt

![Face detection](/img/20180603/Face_detection.png)*Face detection*

Ví dụ về một problem phức tạp: đọc chữ trong ảnh 

![Scene_text](/img/20180603/scene_text.png)*Scene text recognition*

#### 2. Thu thập dữ liệu (Data Gathering)

Sau khi đã định nghĩa vấn đề xong, chúng ta sẽ bắt đầu tiến hành thu thập nguồn dữ liệu tương ứng. Ví dụ như ở trong problem nhận diện khuôn mặt ở trên, chúng ta sẽ phải thu thập dữ liệu là khuôn mặt con người. Số lượng dữ liệu càng nhiều càng tốt. Ở một số nơi con số có thể lên tới hàng triệu.

#### 3. Lọc dữ liệu (Data Parsing)

Dữ liệu 