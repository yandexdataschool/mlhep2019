# Learning to pivot

~~~
\subtitle{Machine Learning in High Energy Physics}
\author{Maxim Borisyak}

\institute{National Research University Higher School of Economics}
\usepackage{amsmath}

\usepackage{caption}

\usepackage{algorithm}
\usepackage{algpseudocode}

\DeclareMathOperator*{\E}{\mathbb{E}}

\DeclareMathOperator*{\var}{\mathbb{D}}
\newcommand\D[1]{\var\left[ #1 \right]}

\DeclareMathOperator*{\argmin}{\mathrm{arg\,min}}
\DeclareMathOperator*{\argmax}{\mathrm{arg\,max}}

\newcommand\dmid{\,\|\,}

\setlength{\jot}{12pt}

\newcommand{\KL}[2]{\mathrm{KL}\left(#1 \;\middle\|\; #2\right)}


\newcommand\blfootnote[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup
}
~~~

### Learning to pivot

Pivoted classifier`\footnote{One can also consider a regressor or any other method that is based on likelihood maximization.}` $f$ is a classifier which output does not depend on nuisance parameters $Z$:

~~~equation*
\forall s: \forall z, z': P(f(X) = s \mid Z = z) = P(f(X) = s \mid Z = z')
~~~

Examples:

- legal reasons;
- differences between simulation and real data;
- unobservable nuisance parameters.

~~~
\blfootnote{This presentation is based on \texttt{https://arxiv.org/abs/1611.01046}}
~~~

### Measuring dependencies

Ideally, classifier should be regularized:

~~~equation*
  \mathcal{L}_\mathrm{pivot}(f) = \mathcal{L}(f) + \operatorname{dependency-measure}(f, Z) \to \min
~~~

~~~center
  \Large \textbf{ How dependencies can be measured? }
~~~

### Adversarial training

One way to measure dependency is as predictability of nuisance given output of $f$:

~~~eqnarray*
\operatorname{dependency-measure}(f, Z) &=& -\min_r \mathcal{L}_\mathrm{adv}(r, f(X), Z);\\
\mathcal{L}_\mathrm{adv}(r, f(X), Z) &=& \E_{X, Z} \log P_r(Z \mid f(X)).
~~~

### Learning to pivot with adversarial networks

The final loss function:

~~~eqnarray*
  L_\mathrm{pivoted} = -\frac{1}{N}\sum^N_{i = 1} \log P_f(y_i \mid x_i) +
  \frac{1}{N} \sum^N_{i = 1} \log P_r(z_i \mid f(x_i)) \to \min   
~~~

The training procedure is similar to GAN:
- train adversary $r$;
- make one step for classifier $f$ with fixed $r$.

### Conditional pivoting

Sometimes it is desirable to make a classifier independent from nuisances within each class.

~~~equation*
\forall y: \forall s: \forall z, z': P(f(X) = s \mid Z = z, Y = y) = P(f(X) = s \mid Z = z', Y = y)
~~~

In this case adversary should be parameterized by target, i.e. receive $y$ as an additional input.

## Summary

### Summary

- dependencies can be measured as quality of a trained model;
- it is possible to pivot model with an adversary;
- conditional pivoting is done by adding target as input feature to the adversary.

### Literature

- Louppe G, Kagan M, Cranmer K. Learning to pivot with adversarial networks. InAdvances in neural information processing systems 2017 (pp. 981-990).