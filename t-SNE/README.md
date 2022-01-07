# t-SNE
t-SNE(t-distrbuted stochastical neighbor embedding)은 높은 차원의 feature을 가지는 데이터들을 저차원(2차원)의 space로 embedding 하는 알고리즘의 한 방식입니다. 
Nonlinear Dimension Reduction, Manifold Learning 혹은 Data Visualization의 방법들 중 대표적으로 사용되는 tool입니다.

[Roweiss(2002)](https://cs.nyu.edu/~roweis/papers/sne_final.pdf)에서 처음 고안되었으며, [Maaten(2008)](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)에서 Data Visualization의 한 방법으로 소개되었습니다.

## t distribution
(Student) t 분포는 정규분포 _N(μ,<img src="https://latex.codecogs.com/svg.image?\sigma^2" title="\sigma^2" />)_ 를 따르는 확률분포 _**X**_ 의 평균을 추정하고자 하나 σ의 값을 알지 못하는 경우 사용하는 분포입니다. 기본적인 통계검정인 t-검정에 이용됩니다.
t 분포는 다음과 같이 나타낼 수 있습니다 : 
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?T=\frac{Z}{V/\nu}" title="T=\frac{Z}{V/\nu}" />
</p>
여기서 T는 t 분포, Z는 표준정규분포, V는 자유도 ν 인 카이제곱분포를 말합니다. 정규분포를 따르는 확률분포의 n개의 샘플을 생각했을 때, Sample Variance가 자유도 _n-1_ 인 카이제곱분포를 따른다는 것을 이용하면 위에서 언급한 t-검정의 식을 유도할 수 있습니다.

t 분포의 확률밀도함수(pdf) 는 다음과 같이 표현할 수 있습니다 :
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?f(x)&space;\propto&space;&space;\left(1&plus;\frac{x^2}{\nu}\right)^{-\left(\frac{\nu&space;&plus;1}{2}\right)}" title="f(x) \propto \left(1+\frac{x^2}{\nu}\right)^{-\left(\frac{\nu +1}{2}\right)}" />
</p>
여기서 자유도 ν를 무한으로 보내면, 표준정규분포로 converge in distribution 할 것을 확인할 수 있습니다.

## Algorithm
t-SNE의 목표는 고차원의 원본 데이터셋 _X_ 와 데이터 구조가 같은 낮은차원에서의 집합  _Y_ 를 찾는 것입니다. 이때 데이터 구조를 나타내는 표현으로 "similarity"를 사용합니다.

Similarity는 두 점 _x1,x2_ 들 사이의 조건부확률을 통해 값을 매깁니다. Roweiss(2002)에서는 가우시안분포를 사용했는데, 두점 중 조건으로 설정한 점을 가우시안분포의 mean으로 잡고, 조건부확률을 계산합니다. 가우시안분포의 분산 <img src="https://latex.codecogs.com/svg.image?\sigma_i" title="\sigma_i" />는 "perplexity"가 사용자가 미리 정해둔 값 (5~50)에 맞도록 binary search를 진행하여 결정하게 됩니다. "perplexity"는 데이터의 엔트로피에 관련된 값으로, 자세한 내용은 Maaten(2008)의 4p를 참고하시기 바랍니다.
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?p_{j|i}&space;=&space;\frac{exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k&space;=&space;1}^n&space;exp(-||x_i-x_k||^2/2\sigma_i^2)}&space;=&space;\frac{exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k\ne&space;i}&space;exp(-||x_i-x_k||^2/2\sigma_i^2)}" title="p_{j|i} = \frac{exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k = 1}^n exp(-||x_i-x_k||^2/2\sigma_i^2)} = \frac{exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k\ne i} exp(-||x_i-x_k||^2/2\sigma_i^2)}" />
</p>

위처럼, 원본 데이터셋 _X_ 의 similarity를 구했듯이, _Y_ 에서의 similarity도 계산하게 됩니다.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?q_{j|i}=\frac{exp(-||y_i-y_j||^2)}{\sum_{k\ne&space;i}exp(-||y_i-y_k||^2)}" title="q_{j|i}=\frac{exp(-||y_i-y_j||^2)}{\sum_{k\ne i}exp(-||y_i-y_k||^2)}" />
</p>

SNE의 목표가 _X_ 와 _Y_ 가 같은 페어에서 같은 similarity을 가지길 원하기 때문에, 비용함수로 Kullback-Leibler divergence를 이용하게 됩니다. 
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?C&space;=&space;\sum_i&space;KL(P_i||Q_i)&space;=&space;\sum_i&space;\sum_j&space;p_{j|i}&space;log&space;\frac{p_{j|i}}{q_{j|i}}" title="C = \sum_i KL(P_i||Q_i) = \sum_i \sum_j p_{j|i} log \frac{p_{j|i}}{q_{j|i}}" />
</p>
