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

# use_math: true

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


### Introduction: Generative Models

생성모델의 목적은 관찰한 sample x에 대해서, x의 분포도를 알아내는 것이다. 이를 이용해 **새로운 데이터들을 sampling을 통해 생성**할 수도 있고, 어떤 데이터에 대한 likelihood를 계산할 수도 있음.

여러가지 방식의 생성모델들: 
- GAN: adversarial한 방식을 이용해 분포도 학습
- autoregressive model, VAE: likelihood-based 모델 
- Score-based generative models

이중에서 diffusion은 likelihood-based모델과 score-based모델 두가지의 관점으로 해석이 가능하고, 먼저 이번 글에서는 likelihood 관점의 해석을 분석해보자.  


### ELBO, VAE and Hierarchical VAE

어떤 데이터가 있으면, 이를 몇가지의 특징을 이용해 표현하거나, 반대로 이러한 특징들을 이용해 데이터를 생성할 수 있다. 이러한 특징들이 latent variable **z**에 해당한다. 저차원의 latent들로 데이터를 표현할 수 있으면, latent들이 semantic할 것이라고 생각할 수 있다. 

우리가 관찰한 데이터 **x**와 latent variable **z**의 joint distribution인 $p(x,z)$를 생각해 보자. 이를 이용해 우리가 구하고자 하는 p(x)를 두가지 방식으로 나타낼 수 있다:

$$
\begin{align}
    p(x) = \int p(x,z)dz
\end{align}
$$
$$
\begin{align}
    p(x) = \cfrac{p(x,z)}{p(z|x)}
\end{align}
$$

하지만, 위의 식들을 이용해 p(x)를 직접 구하기는 어려움. (1)의 경우 모든 latent **z**에 대해서 joint distribution을 구하기 어렵고, (2)의 경우 ground truth latent encoder $p(z|x)$를 얻을 수 없기 때문이다.  
 - latent encoder $p(z|x)$: sample x가 주어졌을때의 z의 분포  

생성모델의 목표는 x의 분포를 알아내는 것이고, 이는 결국 $\log{p(x)}$를 최대화 시켜야 함. 

p(x)는 likelihood이다. likelihood(가능도, 우도)와 probability(확률)은 비슷한 개념이지만, 확실히 차이가 있는 개념이다. 

> **probability**: 확률분포를 고정시켰을 때, 그 분포에 따르면 이 값이 나올 확률이 얼마나 되는가?  
> **likelihood**: 관찰한 값들을 토대로 이 값들이 어떤 확률분포에서 생성되었을까?
  
결국 log-likelihood인 $\log{p(x)}$을 최대화시켜야 하는 것이 생성모델의 목표이다. 

ELBO를 식으로 표현하면 다음과 같다: 

{{< rawhtml >}}
$$
\begin{align}
    \log{p(x)} \geq \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg]
\end{align}
$$
{{< /rawhtml >}}

- $q_\phi(z|x)$: flexible approximate **variational** distribution with parameter $\phi$ = true posterior $p(z|x)$를 추정하는 parameterizable model
> **variational**: 변분. (추후 정리 예정)   
> **posterior**: 관측 값이 주어졌을 때, 구하고자 하는 대상이 나올 확률   
> **prior**: 구하고자 하는 대상 자체에 대한 확률 

$\log{p(x)}$를 직접적으로 계산해 최대화 하는 대신, Lower bound를 최대화하는 방향으로 학습을 진행한다. 이 과정은 $q_\phi(z|x)$가 $p(z|x)$를 추정하도록 $\phi$를 학습하는 것으로 볼 수 있다. 

위의 부등식을 유도하는 방식은 두가지이다. 첫번째로, Jensen's Inequality를 이용하는 방식은 다음과 같다: 

{{< rawhtml >}}
$$
\begin{align}
    \log{p(x)} &= \log{\int p(x,z)dz} & \\
    &= \log{\int\cfrac{p(x,z)q_\phi(z|x)}{q_\phi(z|x)}dz} \\
    &= \log{\mathbb{E}_{q_\phi(z|x)}\bigg[\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg]\\
    &\geq \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg] & \leftarrow \mathrm{Jenson's \ Inequality}
\end{align}
$$
{{< /rawhtml >}}

이 방식은 수식에 대한 semantic한 의미를 찾기가 어렵다. 대신에, 두번째 방식으로 유도하는 과정은 다음과 같다: 

{{< rawhtml >}}
$$
\begin{align}
    \log{p(x)} &= \log{p(x)}\int{q_\phi(z|x)dz} \\ 
    &= \int{q_\phi(z|x)(\log{p(x)})dz} \\ 
    &= \mathbb{E}_{q_\phi(z|x)}[\log{p(x)}] \\ 
    &= \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{p(z|x)}}\bigg] \\ 
    &= \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)q_\phi(z|x)}{p(z|x)q_\phi(z|x)}}\bigg] \\
    &= \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg] + \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{q_\phi(z|x)}{p(z|x)}}\bigg] \\ 
    &= \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg] + D_{KL}(q_\phi(z|x)||p(z|x)) \\ 
    &\geq \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg]
\end{align}
$$
{{< /rawhtml >}}

> (13) -> (14): KL Divergence의 정의.  
> (14) -> (15): KL Divergence는 항상 0보다 크다.  

마지막 부분을 보면, $q_\phi(z|x)$가 $p(z|x)$의 분포도가 유사해질 수록 KL Divergence항이 0에 가까워진다. 근데, $p(z|x)$을 모르기 때문에 KL Divergence항을 직접 최소화하는 것은 어렵다. 대신에, ELBO항을 최대화 시키도록 parameter $\phi$를 optimize하는 것은 가능하다. 

$\log{p(x)}$은 parameter $\phi$에 영향을 받지 않기 때문에 값이 변하지 않는다. 결국 ELBO 항을 최대화 시키는 과정이 KL Divergence항을 최소화 시키게 되고, 이는 $q_\phi(z|x)$가 $p(z|x)$의 분포도가 유사해지는 방향으로 최적화가 이루어진다. 

### Variational Autoencoders (VAE)

<img src="/images/diffusion/2024-03-12-understanding-diffusion-model/figure1.png" width="50%"/>

생성모델의 한 종류로, encoder(**$q_\phi(z|x)$**)을 통해 sample data를 latent vector로 변환하고, 이를 다시 decoder(**$p_\theta(x|z)$**)을 통해 본래의 sample data를 복원하는 과정을 통해 학습을 진행한다. 그리고 latent vector을 조절해 decoder을 거쳐 새로운 데이터를 sampling 할 수 있다. 이때, VAE는 ELBO를 직접 최대화하는 방식으로 optimize한다. ELBO항을 정리하면 다음과 같다: 

{{< rawhtml >}}
$$
\begin{align}
    \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(x,z)}{q_\phi(z|x)}}\bigg] &= \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p_\theta(x|z)p(z)}{q_\phi(z|x)}}\bigg] \\ 
    &= \mathbb{E}_{q_\phi(z|x)}[\log{p_\theta(x|z)}] + \mathbb{E}_{q_\phi(z|x)}\bigg[\log{\cfrac{p(z)}{q_\phi(z|x)}}\bigg] \\ 
    &= \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log{p_\theta(x|z)}]}_{\footnotesize\mathrm{reconstruction \ term}} + \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\footnotesize\mathrm{prior \ matching \ term}} 
\end{align}
$$
{{< /rawhtml >}}

> **reconstruction term**: 식 그대로 말로 풀면, "z의 분포가 $q_\phi(z|x)$ 일때의 $p_\theta(x|z)$의 기댓값"이다. 해석해보면 latent vector z로 변환했을 때, 이 값을 이용해 다시 x가 복원이 될 확률에 대한 기댓값이다. 결국, parameter $\phi$는 latent vector z를 잘 생성하도록, parameter $\theta$는 z에서 x로 잘 복원하도록 optimize하는 항으로 볼 수 있다.  
> **prior matching term**: encoder $q_\phi(z|x)$가 latent prior $p(z)$와 얼마나 유사하냐를 의미한다. 이 항을 최소화 시키려면, $q_\phi(z|x)$가 $p(z)$의 분포와 유사하도록 optimize해야 한다. 

위 식에서 볼 수 있듯이, VAE에서는 ELBO를 최대화하는 과정은 reconstruction term을 크게 하고, prior matching term을 작게 하는 parameter $\phi$, $\theta$를 optimize하는 것과 같다. 

VAE는 일반적으로 encoder($q_\phi(z|x)$)은 multivariate Gaussian으로 모델링 하고, prior($p(z)$)은 standard multivariate Gaussian으로 선택한다. 이를 수식으로 나타내면 다음과 같다: 
{{< rawhtml >}}
$$
\begin{align}
    q_\phi(z|x) &= \mathcal{N}(\mathbf{z}; \mathbf{\mu_\phi(x)}, \mathbf{\sigma_\phi^2(x)I}) \\
    p(x) &= \mathcal{N}(z; \mathbf{0}, \mathbf{I})
\end{align}
$$
{{< /rawhtml >}}

이 수식을 이용하면 ELBO의 KL Divergence항은 직접 계산이 가능하고, reconstruction term의 경우 Monte Carlo Estimation을 이용해 계산할 수 있다. 위의 objective 식을 변경하면 다음과 같다:   
{{< rawhtml >}}
$$
\begin{align}
\begin{split}
    \argmax_{\phi, \theta}\mathbb{E}_{q_\phi(z|x)}[\log{p_\theta(x|z)}] - D_{KL}(q_\phi(z|x)\ ||\ p(z)) \\
    \approx \argmax_{\phi, \theta}\sum_{l=1}^L{\log{p_\theta(x|z^{(l)})}} - D_{KL}(q_\phi(z|x)\ ||\ p(z))
\end{split}
\end{align}
$$
{{< /rawhtml >}}  

- ${[z^{(l)}]}_{l=1}^L$: 모든 관측값 $x$에 대해서 분포 $q_\phi(z|x)$에서 sampling된 값 

그런데, 이렇게 stochastic sampling을 이용해 값을 추정하게 되면 미분 불가능하게 되어 backpropagation이 되지 않아 학습이 불가능하다. 이를 해결하기 위해 분포 $q_\phi(z|x)$를 reparameterization trick을 이용해 다음과 같이 deterministic한 함수 식으로 변경하여 사용한다:  
{{< rawhtml >}}
$$
\begin{align}
    \mathbf{z} = \mathbf{\mu_\phi(x)} + \mathbf{\sigma_\phi(x)\ \odot\ \epsilon}\quad\text{with}\  \epsilon\sim\mathcal{N}(\mathbf{\epsilon; 0, I})
\end{align}
$$
{{< /rawhtml >}}

- $\odot$: element-wise product

### Hierarchical Variational Autoencoders (HVAE)

<img src="/images/diffusion/2024-03-12-understanding-diffusion-model/figure2.png" width="80%"/>


VAE에서 확장하여 여러 단계의 latent variable을 갖는 모델이다. 이때 latent variable($z_t$)이 직전 단계의 latent($z_{t-1}$)에만 영향을 받으면 Markovian HVAE (MHVAE)라고 부른다. joint distribution과 posterior을 식으로 나타내면 다음과 같다: 
{{< rawhtml >}}
$$
\begin{align}
    p(x,z_{1:T}) &= p(z_T)p_\theta(x|z_1)\prod_{t=2}^Tp_\theta(z_{t-1}|z_t) \\
    q_\phi(z_{1:T}|x) &= q_\phi(z_1|x)\prod_{t=2}^Tq_\phi(z_t|z_{t-1})
\end{align}
$$
{{< /rawhtml >}}

그렇게 되면 ELBO는 다음과 같이 표현할 수 있다: 
{{< rawhtml >}}
$$
\begin{align}
    \log{p(x)} &= \log{\int{p(x,z_{1:T}})dz_{1:T}} \\
    &= \log{\int{\cfrac{p(x,z_{1:T})q_\phi(z_{1:T}|x)}{q_\phi(z_{1:T}|x)}dz_{1:T}}} \\
    &= \log{\mathbb{E}_{q_\phi(z_{1:T}|x)}\bigg[\cfrac{p(x,z_{1:T})}{q_\phi(z_{1:T}|x)}\bigg]} \\ &\geq \mathbb{E}_{q_\phi(z_{1:T}|x)}\bigg[\log{\cfrac{p(x,z_{1:T})}{q_\phi(z_{1:T}|x)}}\bigg] \\
\end{align}
$$
{{< /rawhtml >}}

### Variational Diffusion Models (VDM)

<img src="/images/diffusion/2024-03-12-understanding-diffusion-model/figure3.png" width="80%"/>

VDM은 위에서 설명한 Markovian HVAE에 세가지 조건이 붙은 모델이다:
- latent의 차원이 data의 차원과 같다 (=shape 같음)
- latent encoder은 linear Gaussian model이다. 
- latent encoder의 Gaussian model의 hyperparameter은 timestep $t$마다 다르며, 최종 latent인 $x_T$는 standard Gaussian분포를 따른다. 

첫번제 제약조건에서, 우리는 이제 original data와 latent를 모두 $x_t$로 표현할 수 있다. $t=0$일때 true data, $t \in [1, T]$이면 latent variable. 또한, 두번째 조건에서 각각의 Gaussian encoder의 평균($\mu_t(x_t) = \sqrt{\alpha_t}x_{t-1}$)과 표준편차($\sum_t{x_t} = (1-\alpha_t)\mathbf{I}$)를 hyperparameter로 설정하거나 learnable parameter로 설정할 수 있다. 세번째 조건에서, 마지막 latent인 $p(x_T)$가 standard Gaussian임을 확인할 수 있다. encoder와 joint distribution을 식으로 표현하면 다음과 같다: 
{{< rawhtml >}}
$$
\begin{align}
    q(x_t|x_{t-1}) &= \mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)\mathbf{I}) \\
    p(x_{0:T}) &= p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}|x_t) \\
    \text{where}\nonumber \\
    p(x_T) &= \mathcal{N}(x_T;\mathbf{0, I})
\end{align}
$$
{{< /rawhtml >}}

encoder의 경우 parameter $\phi$가 더이상 없기 때문에 학습이 필요하지 않는다. sampling의 경우 Gaussian noise $p(x_T)$에서 $p_\theta(x_{t-1}|x_t)$을 이용해 denoising을 거쳐 $x_0$을 얻을 수 있다. 
HVAE와 마찬가지로, VDM도 ELBO를 최대화함으로써 모델을 최적화할 수 있다: 
{{< rawhtml >}}
$$
\begin{align}
    \log{p(x)} &= \log{\int{p(x_{0:T}})dx_{1:T}} \\
    &= \log{\int{\cfrac{p(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)}dx_{1:T}}} \\
    &= \log{\mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\cfrac{p(x_{0:T})}{q(x_{1:T}|x_0)}\bigg]} \\ 
    &\geq \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_{0:T})}{q(x_{1:T}|x_0)}}\bigg] \\
    &=\mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}|x_t)}{\prod_{t=1}^Tq(x_t|x_{t-1})}}\bigg] \\
    &= \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_T)\textcolor{red}{p_\theta(x_0|x_1)}\prod_{t=\textcolor{red}{2}}^Tp_\theta(x_{t-1}|x_t)}{\textcolor{red}{q(x_T|x_{T-1})}\prod_{t=1}^{\textcolor{red}{T-1}}q(x_t|x_{t-1})}}\bigg] \\
    &= \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_T)p_\theta(x_0|x_1)\prod_{t=1}^{T-1}p_\theta(x_{t}|x_{t+1})}{q(x_T|x_{T-1})\prod_{t=1}^{T-1}q(x_t|x_{t-1})}}\bigg] \\
    &= \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_T)p_\theta(x_0|x_1)}{q(x_T|x_{T-1})}}\bigg] + \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\prod_{t=1}^{T-1}\cfrac{p_\theta(x_t|x_{t+1})}{q(x_t|x_{t-1})}}\bigg] \\
    &= \mathbb{E}_{q(x_{1:T}|x_0)}[\log{p_\theta(x_0|x_1)}] + \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_T)}{q(x_T|x_{T-1})}}\bigg] + \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\sum_{t=1}^{T-1}\log{\cfrac{p_\theta(x_t|x_{t+1})}{q(x_t|x_{t-1})}}\bigg] \\
    &= \mathbb{E}_{q(x_{1:T}|x_0)}[\log{p_\theta(x_0|x_1)}] + \mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p(x_T)}{q(x_T|x_{T-1})}}\bigg] + \sum_{t=1}^{T-1}\mathbb{E}_{q(x_{1:T}|x_0)}\bigg[\log{\cfrac{p_\theta(x_t|x_{t+1})}{q(x_t|x_{t-1})}}\bigg] \\
    &= \mathbb{E}_{q(x_{1:T}|x_0)}[\log{p_\theta(x_0|x_1)}] + \mathbb{E}_{q(x_{T-1}, x_T|x_0)}\bigg[\log{\cfrac{p(x_T)}{q(x_T|x_{T-1})}}\bigg] + \sum_{t=1}^{T-1}\mathbb{E}_{q(x_{t-1}, x_t, x_{t+1}|x_0)}\bigg[\log{\cfrac{p_\theta(x_t|x_{t+1})}{q(x_t|x_{t-1})}}\bigg] \\
    &= \underbrace{\mathbb{E}_{q(x_{1:T}|x_0)}[\log{p_\theta(x_0|x_1)}]}_{\footnotesize\mathrm{reconstruction \ term}} - \underbrace{\mathbb{E}_{q(x_{T-1}|x_0)}[D_{KL}(q(x_T|x_{T-1})\ ||\ p(x_T))]}_{\footnotesize\mathrm{prior \ matching \ term}} \\
    &\qquad- \sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q(x_{t-1}, x_{t+1}|x_0)}[D_{KL}(q(x_t|x_{t-1})\ ||\ p_\theta(x_t|x_{t+1}))]}_{\footnotesize\mathrm{consistency \ term}}\nonumber \\
\end{align}
$$
{{< /rawhtml >}}  
