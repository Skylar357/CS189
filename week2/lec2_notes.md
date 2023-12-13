### Lecture 2. Maximum likelihood estimator
#### ML: main abstract ideas
+ Training data set: $D = \{(x_i, y_i)\}_{i=1}^N$, where $x_i \in R^D$, $y_i \in R$ or $y_i \in \{-1, 1\}$.
+ Model class: $f(x | w, b) = w^T x + b$
+ Loss Function: $L(a, b) = (a - b)^2$ Squared loss
+ Learning Objective (Optimization Problem): $argmin_{w,b} \sum_{i=1}^{N} L\left(y_i, f(x_i | w, b)\right)$

#### Multivariate distributions
#### The basic set-up of MLE
+ Given data $D = \{x_i\}_{i=1}^N$ for $x_i \in R^d$
+ Assume a set (family) of distribution on $R^d$, $\{p_{\theta}(x)|\theta \in \Theta \}$
+ Assume $D$ contains samples from one of the these distributions:
$$x_i \tilde p_{\hat{\theta}}(x)$$
+ This assumes that each element of $D$ is identically and independently distributed (iid)

#### Some properties of MLE
+ The MLE is a consistent estimator: meaning thatasweget more and more data (drawn from one distribution in our family), then we converge to estimating the true value of for .
+ The MLE is statistically efficient:it’s making good use of the data available to it ( “least variance” parameter estimates).
+ The value of $p(D|\theta_{MLE})$ is invariant to re-parameterization.
+ MLE can still yield a parameter estimate even when the data were not generated from that family (phew & caveat emptor).

#### Relationship between likelihood, cross-entropy
+ The cross-entropy is a term from information theory.
+ To understand the connection between MLE and
maximizing the cross-entropy, we need to know some concepts from information theory:
  1. Entropy: 
  2. Cross-entropy
  3. KL-divergence (relative entropy).

#### Entropy: a measure of expected surprise
Think about a flipping a coin once, and how surprised you would be at observing a head.\
+ The “surprise” of observing that a discrete random variable
takes on value $k$ is 
$$log \frac {1} {P(Y = k)} = -log(P(Y = k))$$
+ As $P(Y = k) \rightarrow 0$, the surprise of observing $k$ approaches $\infty$
+ As $P(Y = k) \rightarrow 1$, the surprise of observing $k$ approaches $0$
+ The entropy of the distribution of $Y$ is the _expected surprise_:
$$H(Y) = E_{Y} \{-log P (Y = k)\} = \sum_k P(Y=k)logP(Y = k)$$

#### Entropy example: flipping a coin
$$H(Y) = - \sum_{i=1}^K P(Y = y_i)\log_2 P(Y = y_i)$$
$$P(Y=t) = 5/6,~P(Y=f) = 1/6$$
$$H(Y) = -5/6\log_2 5/6 - 1/6\log_2 1/6 = 0.65$$

#### Entropy of a random variable $Y$:
"High Entropy"
 + $Y$ is from a uniform like distribution
 + Flat histogram
 + Values sampled from it are less predictable

"Low Entropy"
 + $Y$ is from a a varied (peaks and valleys) distribution
 + Histogram has many lows and highs
 + Values sampled from it are more predictable

#### From Entropy to Relative Entropy
+ Also called the Kullback-Leibler (KL) Divergence.
+ Measures how much one distribution diverges from another. 
+ For discrete probability distributions, and , it is defined as:
$$D_{KL}(P||Q) = \sum_{x}P(x)\ln \frac {P(x)}{Q(x)}$$
+ Not a true distance metric because not symmetric in 
$$D_{KL}(P||Q) \neq D_{KL}(Q||P)$$
__Properties of KL Divergence__\
1. $KL(p||q) \geq 0$
2. $KL(p||q) = 0$ if and only if $p=q$

#### From Relative Entropy to Cross-Entropy 
$$\begin{aligned}
D_{KL} &= \sum_x P(X) \log \frac {P(X)} {Q(x)} \\
       &= E_{P(x)}\left[\log \frac {1} {Q(X)}\right] -E_{P(x)}\left[\log \frac {1} {P(X)}\right]\\
       &= H(P, Q) - H(P)
\end{aligned}$$
where $H(P, Q)$ is cross-entropy and $H(P)$ is entropy.
+ Consider data $D$ where $x_i  \hat{p}_{data}$ and a model with params $\theta, p(x|\theta)$.
+ If minimizing the KL divergence (instead of MLE)
$$\begin{aligned}
argmin_{\theta}D_{KL}(\hat{p}_{data}||p(x|\theta)) &= argmin_{\theta} H(\hat{p}_{data}, p(x|\theta)) + H(\hat{p}_{data})\\
&= argmaxE_{\hat{p}_{data}\left[\log p(x|\theta)\right]}\\
&= argmax\sum_i^{N}\log p(x_i|\theta)
\end{aligned}$$

+ Performing MLE maximizes the likelihood function.
+ This is equivalent to maximizing the cross-entropy.
+ And equivalent to minimizing the KL-divergence (aka relative entropy).