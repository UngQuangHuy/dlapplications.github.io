---
layout: post
title: Nền tảng của deep learning - Perceptron
subtitle: Deep learning cơ bản (phần 1)
tags: [beginner, perceptron]
math: true
---
# Nền tảng của deep learning - Perceptron
Chúng ta thường nghe báo chí đưa tin rằng deep learning (DL) mô phỏng cách mà các neuron (tế bào thần kinh) trong não chúng ta hoạt động. Cách nói này chỉ đúng một phần vì các mô hình neural network trong deep learning hiện nay chỉ sử dụng những nguyên lý hoạt động đơn giản nhất của các neuron dựa trên các nghiên cứu từ những năm 1940s. Từ đó đến nay các nhà khoa học đã khám phá ra rằng các neuron hoạt động theo những nguyên lý phức tạp hơn thế rất nhiều. Tuy vậy những đột phá mà deep learning đem lại đã cho thấy những hiểu biết dù là đơn giản nhất về bộ não con người có luôn là một nguồn cảm hứng bất tận cho AI. Hôm nay nhóm sẽ giới thiệu tới các bạn mô hình neural network đầu tiên và cũng là nền tảng cho các mô hình phức tạp sau này trong DL. Mô hình đó được gọi là Perceptron.
<br/>Trước hết chúng ta hãy cùng tìm hiểu cách mà Perceptron mô phỏng lại neuron trong não chúng ta. Nếu bạn nào đã học hết cấp 3 ở Việt Nam chắc cũng đã có lần được học về cấu tạo của neuron trong môn sinh học. Đại khái một neuron nó có cấu tạo như thế này. 
![neuron](/img/20180611/bio-neuron.jpg)<br/>
> Hình 1. Cấu tạo neuron sinh học.
> 

Chúng ta được học là một neuron sẽ nhận các tín hiệu điện (electrical signal) có chứa thông tin từ các synapse của một hay nhiều neuron khác thông qua các đuôi gai (dendrit ). Các giá trị tín hiệu điện đầu vào (__input__) sẽ được tích tụ lại trong thân neuron (cell body). Nếu tổng các tín hiệu này vượt quá một ngưỡng (__threshold__) nhất định thì thân neuron sẽ phát ra một tín hiệu điện đầu ra (__output__) truyền qua sợi trục (axon) tới các synapse. Neuron lúc này được gọi là đang __firing__ hay đã __activated__. Tín hiệu điện output này sẽ được truyền sang các neuron khác nhờ vào sự liên kết giữa synapse và dendrit, độ mạnh yếu của các liên kết sẽ quyết định lượng thông tin được truyền sang. Cứ như thế quá trình này khi diễn ra đồng thời giữa một mạng lưới các neuron sẽ tạo nên hoạt động của hệ thần kinh trong bộ não chúng ta. Vào năm 1958, nhà khoa học Frank Rosenblatt đã dựa trên nguyên lý hoạt động trên để đề xuất ra mô hình perceptron như hình dưới.
![perceptron](/img/20180611/perceptron.png)<br/>
> Hình 2. Perceptron-mô hình đơn giản của một artificial neuron (neuron nhân tạo) trong DL.
> 

Trong mô hình trên, hình tròn màu đỏ là đại diện cho thân neuron. Các giá trị đầu vào $$x_1, x_2, ..., x_n$$ (chúng ta sẽ nói sau về đầu vào có giá trị 1) đại diện cho giá trị của các tín hiệu điện truyền tới từ các synapse của các neuron khác. Các input này sẽ được nhân lần lượt với các giá trị $$w_1, w_2, w_3, ..., w_n$$ (được gọi là các *__weights__*) tương ứng trước khi truyền tới thân neuron màu đỏ. Đây là cách perceptron mô phỏng độ mạnh yếu giữa các liên kết synapse-dendrit để điều chỉnh lượng thông tin được truyền sang. Nếu ta gọi tổng của các tín hiệu được tích tụ lại tại neuron cell body là $$z$$ thì ta sẽ có:
>   
$$
z = \sum_{i=1}^{n} w_ix_i = w_1x_1 + w_2x_2 + ... + w_nx_n
$$

Cũng giống như nguyên lý của một neuron sinh học đã nói đến ở trên nếu tín hiệu $z$ này lớn hơn một threshold (giá trị ngưỡng) nhất định thì neuron sẽ vào trạng thái fired (hay activated) và xuất ra một tín hiệu output mà ta sẽ gọi là $h$. Trong DL giá trị threshold và độ mạnh của tín hiệu output sẽ được quyết định thông qua một hàm số cụ thể và các hàm số này được gọi chung là *__activation function__* (bởi vì làm neuron activated). Để đơn giản ta sẽ giả sử như sau:
 >   
$$ h =
\begin{cases}
0\quad nếu \sum w_ix_i < threshold\\
1\quad nếu \sum w_ix_i\geq threshold\\
\end{cases}
$$
    
Nghĩa là nếu tổng input lớn hoặc bằng ngưỡng thì neuron sẽ output 1 còn không thì không output gì cả. Nếu đặt $w_0=-threshold$ thì ta có thể viết lại hàm số trên như sau:
>
$$
h =
\begin{cases}
0\quad nếu \sum w_ix_i + w_0 < 0\\
1\quad nếu \sum w_ix_i + w_0 \geq 0\\
\end{cases}
$$

Do vậy với bất kì giá trị nào của threshold ta cũng có thể tính tổng lượng tín hiệu input $z$ như cũ rồi cộng thêm một giá trị $w_0=-threshold$ mà ta có thể coi như là tích của một input là 1 với weight là $w_0$ như trong hình 2 (input 1 này được gọi là __bias unit__). Sau đó ta so sánh tổng này với 0 để quyết định output $h$.
<br/>Như ta có thể thấy một perceptron như trên có thể dùng trong bài toán __binary classification__ (phân loại nhị phân) trong đó ta sẽ dựa trên các giá trị input để quyết định sẽ chia đối tượng vào 1 trong 2 nhóm đã định sẵn. Ví dụ như là dựa trên đặc điểm của đối tượng là 1 cái email, khi đó đặc điểm có thể là 10 từ ngữ xuất hiện nhiều nhất trong email. Sử dụng các đặc điểm này làm input cho perceptron để quyết định xem email đó là spam (output 1) hay không phải spam (output 0). Do đó, việc cho perceptron học nghĩa là tìm ra những giá trị phù hợp của các __weight $w$__ sao cho perceptron có thể đưa ra output 1 hay 0 một cách chính xác nhất. Cách để tìm ra các giá trị phù hợp này thường là bằng cách sử dụng một thuật toán có tên là __Gradient descent__. Cách cho học perceptron và một số ứng dụng đơn giản của nó sẽ được nhóm giói thiệu trong các bài viết tiếp theo. Chắc hẳn các bạn đã có lần nhìn thấy hình minh họa một mạng neural network như hình dưới và qua bài viết này các bạn đã biết nó chỉ là sự kết hợp của nhiều perceptron.   
![NN](/img/20180611/ANN.jpg)
> Hình 3. Một artificial neural network (ANN) đơn giản với nhiều lớp perceptron. 
> 
 
