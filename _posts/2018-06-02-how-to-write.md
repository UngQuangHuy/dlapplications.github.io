---
layout: post
title: Hướng dẫn post bài
subtitle: Làm thế nào để viết một bài post chuẩn và đẹp
tags: [tutorial]
bigimg: /img/path.jpg
image: /img/hello_world.jpeg
---

Làm theo những bước sau để viết bài

## Những bước để viết bài

    1. Clone repos ở https://github.com/dlapplications/dlapplications.github.io
    2. Tạo một file mởi ở trong thư mục _post với tên file là YYYY-MM-DD-Title.md (YYYY = year, MM=month, DD=day)
    3. Viết bài, chỉnh sửa tag, etc
    4. Push lên trang github . Bạn nào chưa có quyền write thì tạo một pull request, thành viên trong nhóm sẽ pull vào.

## Viết bài như thế nào

Một bài viết nên viết như sau


    ---
    layout: post
    title: Tiêu đề bài viết
    subtitle: Tiêu đề phụ
    image: /img/hello_world.jpeg  (ảnh minh họa bài viết, file ảnh trong thư mục img của repos)
    bigimg: /img/path.jpg (ảnh nền riêng của bài viết, file ảnh trong thư mục img của repos)
    gh-repo: daattali/beautiful-jekyll
    gh-badge: [star, fork, follow]
    tags: [test] : tag của bài viết. sẽ cập nhật về luật tag sau
    ---

    Nội dung bài viết, bằng markdown
    vân vân

## Một số tool dùng để viết bài

    Visual studio code (Markdown plugins)
    Sublime Text (Markdown plguins)
    Atom    (Markdown plguins)
    etc

## Một số tag thông dụng nên dùng

    tutorial :  các bài viết liên quan đến tutorial
    blog
    new
    supervised
    unsupervised
    etc