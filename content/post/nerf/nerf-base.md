---
title: "NeRF 기본 개념 설명"
date: 2023-12-13T17:30:00+00:00
weight: 2
# aliases: ["/first"]
tags: ["nerf"]
# author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
UseHugoToc: true
draft: false

description: ""
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false

cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
# editPost:
#     URL: "https://github.com/<path_to_repo>/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: false # to append file path to Edit link
---

paper: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

앞으로 리뷰하고 공부하게 될 논문들의 기본 베이스 개념인 NeRF에 대해서 간단히 알아봅시다. 

# Neural Radiance Field


논문에서 다루는 view synthesis 테스크는 어떤 물체를 여러 각도로 찍은 사진을 이용하여 새로운 각도에서 물체를 바라본 이미지를 얻어내는 작업입니다. 

MLP를 사용해 2D image를 input으로 활용하여 3D object의 color값과 volume density값을 예측하고, 이를 이용해 novel view image를 얻어내는 모델입니다. 

논문을 이해하기 위해선 camera model, volume rendering, ray tracing등의 컴퓨터 비전 관련 지식들이 필요하기 때문에, 차후에 하나씩 정리해서 추가해 보도록 하고, 이 글에서는 논문에 쓰인 기법들 위주로 설명해 보도록 하겠습니다. 


## Model Pipeline 

please 9트

![NeRF-pipeline](/images/nerf-base/nerf-pipeline.png)

<img src="/images/nerf-base/nerf-pipeline.png">