---
title: "[논문리뷰] Understanding Diffusion Model: The Unified Perspective"
date: 2024-03-12
weight: 1
# aliases: ["/first"]
categories: "[논문 리뷰]"
tags: 
    - "diffusion"

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
hideSummary: true
searchHidden: false

use_math: true

# cover:
#     image: "</images/2024-03-12-understanding-diffusion-model/cover.png>" # image path/url
#     alt: "<alt text 텍스트>" # alt text
#     caption: "<caption 캡션>" # display caption under cover
#     relative: false # when using page bundles set this to true
#     hidden: true # only hide on current single page
# editPost:
#     URL: "https://github.com/<path_to_repo>/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: false # to append file path to Edit link
---
<style>
r { color: Red }
o { color: Orange }
b { color: Blue }
</style>

### Understanding Diffusion Model: The Unified Perspective  
> [[Paper](https://arxiv.org/abs/2208.11970)]  
> Author: Calvin Luo  
> Google Research, Brain Team  
> 25 Aug 2022
---
diffusion 분야 최신 논문을 읽기 전에, 리뷰논문을 공부하며 VAE부터 Diffusion 모델까지 수식과 함께 이해해 봅시다.  

생성모델이라는 개념을 제대로 접한 것은 이 논문이 처음이라, 최대한 제가 이해한 대로 글을 작성했습니다. 

## Introduction: Generative Models

생성모델의 목적은 관찰한 sample x에 대해서, x의 분포도 **p(x)**를 알아내는것 입니다. 이를 이용해 **<r>새로운 데이터들을 sampling을 통해 생성</r>**할 수도 있고, 어떤 데이터에 대한 likelihood를 계산할 수도 있습니다.

여러가지 방식의 생성모델들: 
- GAN: adversarial한 방식을 이용해 분포도 학습
- autoregressive model, VAE: likelihood-based 모델 
- Score-based generative models

이중에서 diffusion은 likelihood-based모델과 score-based모델 두가지의 관점으로 해석이 가능하고, 먼저 이번 글에서는 likelihood 관점의 해석을 분석해 봅시다. 

## ELBO, VAE and Hierarchical VAE


