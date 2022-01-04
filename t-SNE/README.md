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
