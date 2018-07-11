---
layout: post
title: Sơ lược về Transfer Learning I.
hidden: true
bigimg: /img/20180715/title.jpg
tags: [basic]
math: true
---


# Sơ lược về Transfer Learning.
## Part I. Cơ sở cho Transfer Learning.
### Mục lục:
1. [Giới thiệu](#intro)
2. [Các định nghĩa](#first)
    1. Model
    2. Source Tasks và Target Tasks
    3. Transfer Learning
3. [Lợi ích](#second)
    1. Thời gian
    2. Hiệu  quả
4. [Cơ sở](#third)
    1. Featuriser
    2. Fine-tuning
5. [Kết bài](#conclusion)
    1. Ứng dụng
    2. Hạn chế
    3. Ý tưởng


### 1. Giới thiệu <a name="intro"></a>
Bạn quá lười khi phải train lại network từ đầu?

Bạn quá mệt mỏi với việc tạo ra một end-to-end network mới để thực hiện một task gồm có nhiều phần khác nhau?

Bạn không có một bộ GPU mạnh và không muốn đăng ký AWS hay dùng Google Cloud?
Hay bạn không có một bộ dataset lớn trong task mà mình phải thực hiện?

Well, đã đến lúc bạn hướng tới sự giúp đỡ của Transfer Learning, một kỹ thuật đưa đến những network đủ tốt chỉ với lượng dataset nhỏ trên cơ sở những network có sẵn.
Để tận dụng được các pretrained network này là một nghệ thuật được nghiên cứu từ thập niên 90. Khi Lorien Pratt [thực nghiệm](http://papers.nips.cc/paper/641-discriminability-based-transfer-between-neural-networks.pdf) lần đầu năm 1993 và sau đó viết lại nó dưới dạng một lý thuyết toán học (formal analysis) năm 1998.

Đây sẽ là bài thứ nhất của Transfer Learning, giúp cho các bạn có một cái nhìn tổng quan về các mặt của Transfer Learning trước khi đi vào thực hành ở bài viết kế tiếp ;)

#### Prerequisite - Trước khi học về Transfer Learning, mình khuyến khích các bạn hiểu về:
 Neural Network cơ bản.
 
 Dataset và các bước tạo ra một model.

### 2. Các định nghĩa <a name="first"></a>
#### 1. Model
Chắc hẳn, nhiều bạn cũng đã biết về các model nổi tiếng, được train trên các dataset lớn (MNIST, CIFAR-100, ImageNet, ...) và source code cũng như **Weights** của model được public cho cộng đồng (chủ yếu là trên GitHub).
Chúng ta gọi những Model đi kèm Weights như vậy là một **Pretrained Model**.

Model mới sử dụng một phần hay toàn bộ pretrained model như một phần của nó để học một tasks mới được gọi là **Transfered Model**.

#### 2. Source Tasks và Target Tasks
Những Pretrained Model như vậy thường được train trên một hoặc một vài bộ datasets nhất định, tương thích và cho accuracy cao với một task hoặc nhiều tasks (multi-task deep learning) nào đó mà nó được train. Chúng ta gọi các tasks mà pretrained model đó được train để thực hiện là **source tasks**.

Nhiệm vụ của chúng ta là tạo ra một model mới để thực hiện một hoặc nhiều tasks nào đó. Những tasks cần được thực hiện của model này có thể trùng hoặc không trùng với tasks mà pretrained model được train (thường thì sẽ không trùng), chúng ta gọi tasks này là **target tasks**.

#### 3. Transfer Learning
#### a. Sự cần thiết
Ngày xửa ngày xưa, khi nền văn minh của loài người chưa phát triển, các nhóm người nhỏ sinh sống trong những hang hốc. Khi con người biết trồng trọt, họ chuyển ra những đồng bằng sinh sống và tại đó, họ gặp những bộ tộc khác. Việc hiểu được nhau trở nên khó khăn khi số người càng ngày càng tăng.
Và thế là chúng ta phát minh ra ngôn ngữ, một cách để *truyền đạt* ý nghĩ của mình cho người xung quanh.

Transfer Learning cũng chính là cách để các model truyền đạt cho nhau khả năng mà mỗi model có thể làm được. Một model có thể học trên source tasks nào đó và rồi một phần hoặc toàn bộ pretrained model này được sử dụng cho model khác để model mới đó học trên target tasks nhanh hơn.

#### b. Chính thức
**Transfer Learning** trong Deep Learning là một kỹ thuật mà trong đó:
* Một pretrained model đã được train trên source tasks cụ thể nào đó, khi đó một phần hay toàn bộ pretrained model có thể được tái sử dụng phụ thuộc vào nhiệm vụ của mỗi layer trong model đó.
* Một model mới sử dụng một phần hay toàn bộ pretrained model để học một target tasks và tùy vào nhiệm vụ của mỗi layer mà model mới có thể thêm các layer khác dựa trên pretrained model sẵn có.

### 3. Lợi ích <a name="second"></a>
#### 1. Thời gian
Việc sử dụng Pretrained Model bản thân nó không chỉ giúp giảm thời gian vào việc tạo ra một model mới để thực hiện một target tasks mà dựa trên một source tasks sẵn có, mà còn giảm thời gian train một model từ đầu vì Weights của phần source tasks đã có sẵn.

#### 2. Hiệu quả
Bạn nghĩ mình có thể tạo ra một model mới tốt hơn pretrained model trên cùng source tasks không?

Bạn có muốn tạo một model mới cùng source tasks mà phải có accuracy cao hơn các pretrained state-of-the-art không?

Nếu bạn quá lười để làm điều đó, pretrained model đã cung cấp cho bạn một accuracy cao ngay từ đầu, do đó khi train trên target tasks thì transfered model của bạn sẽ tiếp tục tăng accuracy này thay vì phải bắt đầu từ điểm có accuracy thấp hơn.

![Performance](/img/20180715/0.png)*So sánh tương quan hiệu quả của model train từ đầu và transfered model*

### 4. Cơ sở <a name="third"></a>
#### 1. Featuriser
Trước năm 2012, hầu hết mọi model AI đều được tạo thành bởi 2 stages khác nhau:
* Feature Engineering (Feature Extractor/Hand-crafted Features): những thuật toán cổ điển để tạo ra các feature từ các quan sát của con người về đặc trưng của dataset.
* Classifier/Regressor: các SVMs hoặc các Random Forests để học được các kết quả từ các Features được tạo ra ở bước trên.

![Featuriser](/img/20180715/1.png)*So sánh phương pháp Featuriser của các hình thức cổ điển và Deep Learning hiện đại*

Các model Deep Learning tự nó đã kết hợp 2 stages này lại, hầu hết các layer ở phần đầu của model là các Feature Extractor (Feature Layer) và phần còn lại là để Classify/Regress với dataset. Do đó các Feature Layer này có thể lấy ra được những Feature từ trong dataset mà không cần các thuật toán cụ thể như lúc trước. Kết quả thực tiễn cho thấy hiệu quả của việc extract feature bằng Feature Layer này cao hơn so với các thuật toán cổ điển đó.

Đây là nền tảng của Transfer Learning: chúng ta có thể sử dụng những Feature Layer này để extract feature cho model của chúng ta thay vì phải tạo ra một Feature Layer mới và train lại từ đầu. Có thể nói là thay vì chạy bộ từ đầu đến cuối đường, chúng ta bắt taxi đến đoạn mà taxi không thể đi được thì chúng ta tự đi tiếp. Hãy hình dung sẽ ra sao nếu đoạn đường taxi đi được là gần hết quãng đường mà chúng ta cần đi? :)

#### 2. Fine-tuning
Để sử dụng pretrained model một cách hiệu quả, chúng ta cần phải có 2 điều sau:
* Thêm các layer phù hợp với target tasks của chúng ta, loại bỏ các layer của pretrained model mà chúng ta không dùng đến (việc này chắc các bạn cũng đã biết rồi), những phải khiến cho model **trở nên hiệu quả hơn**, đây là một vấn đề khó (rất rất khó) cần phải có những nghiên cứu chuyên sâu về từng layer và mục đích của chúng.
* Có chiến lược train thật tốt, điều này cũng không phải là dễ, vì nếu các bạn train không tốt thì sẽ làm mất đi tính hiệu quả của pretrained model và do đó giảm khả năng của model mà chúng ta đang train, thậm chí còn tệ hơn là train hết lại từ đầu.

Do đó, **fine-tuning** ra đời để giúp cho các bạn có chiến lược train hiệu quả trên transfered model của mình (điều đầu tiên theo như mình biết thì chưa tổng quát hóa được để tạo ra một kỹ thuật).

Fine-tuning không phải chỉ giúp cho các bạn điều chỉnh weights của transfered model cho phù hợp với target tasks. Nó không phải chỉ là **tinh chỉnh** như dịch nghĩa của fine-tuning mà xa hơn đó, nó đưa ra cách tối ưu để train cả phần pretrained model và phần mới trong transfered model nhằm đạt được accuracy cao trên target tasks, khiến cho 2 phần fit với nhau hoàn chỉnh thành một model mới.

Dưới đây là các chiến lược thường dùng:

![Featuriser](/img/20180715/2.jpg)*Phân loại chiến lược Fine-tuning*

* Khi dataset cho target tasks **lớn và tương tự** với dataset cho source tasks: đây là trường hợp lý tưởng, khi bạn có thể dùng weights của pretrained model để khởi tạo cho phần pretrained, sau đó train cả transfered model hay chỉ với phần được thêm vào, tùy bạn.
* Khi dataset cho target tasks **nhỏ và tương tự** với dataset cho source tasks: vì dataset là nhỏ, nếu train lại phần pretrained sẽ dẫn đến overfitting, do đó chúng ta chỉ train những layer được thêm vào với weights khởi tạo cho pretrained như trên.
* Khi dataset cho target tasks **lớn và khác biệt** với dataset cho source tasks: bởi vì dataset của chúng ta có sự khác biệt nên khi dùng weights từ pretrained model sẽ làm giảm accuracy vì sự khác biệt trong tasks và dataset, nhưng cũng chính vì dataset lớn nên việc train toàn bộ transfered model từ đầu là hiệu quả nhất, giúp cho model thích nghi tốt hơn với dataset này.
* Khi dataset cho target tasks **nhỏ và khác biệt** với dataset cho source tasks: đây là trường hợp khó khăn nhất, điều mà bạn nên làm có thể là:
  + Can thiệp vào pretrained model, thay thế những pretrained layer xa input để thích nghi với dataset mới (những high-level feature sẽ thay đổi vào các low-level feature đã được lấy từ các layer trước đó) nhưng **không được train các layer gần input của pretrained** vì dataset nhỏ sẽ không thể train được các layer này hiệu quả và các layer này chỉ extract các feature tổng quát từ dataset, sẽ không ảnh hưởng đến target task.
  + Tham khảo ý kiến chuyên gia hay tiền bối để biết thêm phương pháp.

### 5. Kết bài <a name="conclusion"></a>
#### 1. Ứng dụng
Transfer Learning mang đến những model mới với độ chính xác cao trong thời gian ngắn, hầu hết các model dùng transfer learning được sử dụng trong các nghiên cứu về Computer Vision (CV), chú trọng vào việc extract các feature từ ảnh hoặc video một cách hiệu quả như một cách thay thế cho các phương pháp cũ (AKAZE, ORB, BRISK, ...) và kết hợp những ý tưởng mới để tận dụng các feature này (Object Detection, Object Recognition, Human Pose Estimation, ...)

Nhưng không phải là Natural Language Processing (NLP) không sử dụng Transfer Learning, thực tế thì: nếu CV dùng Convolutional Network để extract feature từ ảnh thì NLP dùng Word Embeddings như một cách để extract các feature từ các từ ngữ thành những vectors. Hiệu quả thực tiễn của Word Embeddings [cao hơn](https://www.datacamp.com/community/tutorials/lda2vec-topic-model) hẳn one-hot encodings về khả năng đại diện thông tin.

#### 2. Hạn chế?
Transfer Learning không phải một kỹ thuật dễ sử dụng, nếu bạn sai sót trong quá trình transfer architecture của pretrained hay thêm/bớt không đúng layer thì accuracy sẽ **thấp không tưởng tượng được**, khi đó bạn sẽ phải transfer lại từ đấu.

Bạn chỉ có thể transfer learning khi mà có pretrained model liên quan trực tiếp đến target tasks của bạn thôi, không phải pretrained model nào cũng có thể dùng để transfer vào target tasks mà bạn mong muốn được.
Trước khi dùng Transfer Learning, cũng như bao nhiêu hướng đi khác khi học tập cũng như nghiên cứu, bạn phải xác định rằng:
* Liệu có cần thiết phải transfer learning không?
* Chọn pretrained model nào là tốt nhất?
* Chọn dataset như vậy có phù hợp với pretrained model không?
* Chọn cách fine-tune nào là hiệu quả nhất?
* Liệu sau khi train thì model có accuracy cao hơn bình thường không?
* Vâng Vâng và Mây Mây ...

Như các bạn có thấy, rất nhiều thứ để suy nghĩ trước khi transfer learning vì nhiều lý do khác nhau (Không phải source code nào cũng chạy được hay cho ra accuracy như trong paper? Source code thì public, nhưng bạn phải train để có pretrained model, vậy mục đích của transfer learning đang ở nơi nào? Reposistory cung cấp đầy đủ source code, pretrained nhưng bạn phải install ngôn ngữ mới để chạy và khi transfer learning thì bạn phải code trên ngôn ngữ đó, vì vậy phải học ngôn ngữ mới? ... đủ thứ vấn đề bủa vậy bạn khi bạn định dùng pretrained model).

> Vậy nên là chúc các bạn tìm được pretrained model như ý nhé ;)

#### 3. Ý tưởng
Việc sử dụng pretrained model là một bước tiến lớn để những người đi sau tiếp bước những thành quả của các bậc tiền bối, tận dụng những pretrained model sẵn có để tạo ra những model mới phục vụ cho các target tasks cụ thể hơn, mang tính ứng dụng thực tiễn hơn.

Đó không phải là sự sao chép ý tưởng, bản thân người tạo ra pretrained model đó public thành công của họ là vì hy vọng những người theo sau có thể tìm được những ích lợi từ các model đó, hay ít nhất là dùng nó để giải quyết các công việc của họ. 

Việc nghiên cứu khoa học, đưa ra những ý tưởng mới thì quan trọng nhất là ***không làm lại những gì đã được làm rồi mà không làm tốt hơn được*** vì thời gian sẽ không cho phép sự lãng phí như vậy xảy ra. Đặc biệt là trong Deep Learning, một ngành phát triển nhanh đến chóng mặt hiện nay, những ý tưởng mình nghĩ ra chắc gì đã chưa có ai làm? Deep Learing lan tỏa đến mọi lĩnh vực, vì thế cái quan trọng là sử dụng những prior work sẵn có để tạo nên một model mới tốt hơn, vì chính việc này đã rất khó khăn và tốn thời gian rồi chứ không nói đến nghiên cứu lại từ đầu mọi thứ.

References:

[A Gentle Introduction to Transfer Learning for Deep Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

[Transfer learning & The art of using Pre-trained Models in Deep Learning](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)

[Transfer Learning: Leverage Insights from Big Data](https://www.datacamp.com/community/tutorials/transfer-learning)

[Wikipedia](https://en.wikipedia.org/wiki/Transfer_learning)

Part II sẽ là một bài thực hành về Transfer Learning thú vị đang chờ các bạn, hãy cùng đón xem nhé ;)

Hải Đăng