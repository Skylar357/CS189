### Lecture 1. Introduction

#### E.g. of learning problems
#### Types of problems
+ Classification problems, where the output is a label from a finite set (in the simplest case, just binary).
+ Regression problems, where the output is real-valued
+ Ranking problems, such as an internet query for a document relevant to a search term

#### The machine learning approach to object recognition
$$
\begin{matrix} 1 & 1 & 1 & 1\\ 0 & 5 & 1 & 0 \\ 5 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{matrix}
$$
feature vec: $\{1, 1, 1, 1, 0, 5, 1, 0, 5, 1, 0, 0, 1, 0, 0, 0\}^T$

#### How to classify  a new point?
+ Nearest neighbor rule
+ Linear classifier rule: 
  decision boundary: $\boldsymbol{w} \cdot \boldsymbol{x} + b = 0$
  $\boldsymbol{w} and b will be learned at training time.$

#### Non-linear decision boundaries
+ Nearest Neighbors
+ Multilayer perceptrons a.k.a Neural Networks

 We can construct a new higher-dimensional feature space where the boundary is linear.
 non-linear boundary: $\{x_1, x_2\}^T$
 $$x_1^2 + x_2^2 = c$$
 linear boundary: $\{x_1, x_2, x_1^2, x_1 x_2, x_2^2\}$

#### Neural Networks
**-Single layer neural network**
$$g(z) = \frac {1} {1 + e^{-z}}$$
$$V_j = g(\sum w_{jk} \cdot x_k)$$
where $k = 1, 2, \ldots, 5$, $j = 1, 2$.\
**-Two layer neural network**
$$V_j= g(\sum w_{jk} \cdot x_k)$$
$$O_i = g(\sum w_{ij} \cdot V_j) = g(\sum_j w_{ij} \cdot g(\sum_k w_{jk}x_k))$$

**Training a neural network** \
**Goal**: Find $w$ such that $O_i$ is as close as possible to $y_i$ \
**Approach**: 
+ Define a loss function $L(w)$ 
+ Compute $\nabla_w L $
+ $w_{now} \leftarrow w_{old} - \eta \nabla_w L$

**Training a single layer neural network**
+ A good choice of loss function is the cross entropy
$$L = -\sum_{input~data} (y_i \ln O_i + (1 - y_i)\ln(1 - O_i))$$
+ We model the activation function g as a sigmoid
$$g(z) = \frac {1} {1 + e^{-z}}$$
+ Finding w reduces to logistic regression!\
  Use **Stochastic gradient descent**

**Training a two layer neural network**
+ We compute the gradient with respect to all the weights: from input to hidden layer, and hidden layer to output layer.
+ We can use stochastic gradient descent as before. The loss function is no longer convex, so we can only find local minima. That may be good enough for many applications.
+ The complexity of computing the gradient in the naïve version is quadratic in the number of weights. The back- propagation algorithm is a trick that enables it to be computed in linear time.
+ We can add a regularization term to penalize large weights; that usually improves the performance.