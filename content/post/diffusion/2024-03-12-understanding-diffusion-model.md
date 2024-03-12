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

생성모델의 목적은 관찰한 sample x에 대해서, x의 분포도를 알아내는 것이다. 이를 이용해 **새로운 데이터들을 sampling을 통해 생성**할 수도 있고, 어떤 데이터에 대한 likelihood를 계산할 수도 있음.

여러가지 방식의 생성모델들: 
- GAN: adversarial한 방식을 이용해 분포도 학습
- autoregressive model, VAE: likelihood-based 모델 
- Score-based generative models

이중에서 diffusion은 likelihood-based모델과 score-based모델 두가지의 관점으로 해석이 가능하고, 먼저 이번 글에서는 likelihood 관점의 해석을 분석해보자. 

## ELBO, VAE and Hierarchical VAE

어떤 데이터가 있으면, 이를 몇가지의 특징을 이용해 표현하거나, 반대로 이러한 특징들을 이용해 데이터를 생성할 수 있다. 이러한 특징들이 latent variable **z**에 해당한다. 저차원의 latent들로 데이터를 표현할 수 있으면, latent들이 semantic할 것이라고 생각할 수 있다. 

우리가 관찰한 데이터 **x**와 latent variable **z**의 joint distribution인 p(x,z)를 생각해 보자. 이를 이용해 우리가 구하고자 하는 p(x)를 두가지 방식으로 나타낼 수 있다:

$$
\begin{equation}
    p(x) = \int p(x,z)dz
\end{equation}
$$
$$
\begin{equation}
    p(x) = \cfrac{p(x,z)}{p(z|x)}
\end{equation}
$$

하지만, 위의 식들을 이용해 p(x)를 직접 구하기는 어려움. (1)의 경우 모든 latent **z**에 대해서 joint distribution을 구하기 어렵고, (2)의 경우 GT(ground truth) latent encoder p(z|x)를 얻을 수 없기 때문이다. 
> latent encoder p(z|x): x가 주어졌을때의 z의 분포  
> x를 latent variable **z**로 변환해주는 encoder로 볼 수 있음.

생성모델의 목표는 x의 분포를 알아내는 것이고, 이는 결국 $$\log{p(x)}$$를 최대화 시켜야 함. 

> p(x): likelihood  
> > likelihood(가능도, 우도) <-> probability(확률)  
> > **probability**: 확률분포를 고정시켰을 때, 그 분포에 따르면 이 값이 나올 확률이 얼마나 되는가?  
> > **likelihood**: 관찰한 값들을 토대로 이 값들이 어떤 확률분포에서 생성되었을까?  
> 결국 log-likelihood인 \\(\log{p(x)}\\)을 최대화시켜야함.  
제발 