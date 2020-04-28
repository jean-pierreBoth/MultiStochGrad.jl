
# MultiStochGrad

**WARNING : This is a preliminary version of a Julia package not yet stored in Julia registry**.

## This package provides an implementation of 2 stochastic gradient algorithms: SCSG and SVRG

 The algorithms implemented are described in the following papers:

1. The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    - "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    - "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

2. The SVRG algorithm described in the paper by R. Johnson and T. Zhang
    "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2013).  
    [Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

These algorithms minimize functions given by an expression: 

        f(x) = 1/n ∑ fᵢ(x) where fᵢ is a convex function.

All algorithms alternates some form of large batch computation (computing gradient of many terms of the sum)
and small or mini batches (computing a small number of terms, possibly just one, term of the gradient)
and updating position by combining these global and local gradients.
Further documentation can be found in docs of the Julia package, more in the doc of the Rust crate and at last the reference papers.

## Examples and tests

Small tests consist in a line fitting problem that is taken  from the crate optimisation.

Examples are based on logisitc regression applied to digits MNIST database
(as in the second paper on SCSG).  
The data files can be downloaded from [MNIST](http://yann.lecun.com/exdb/mnist).
The database has 60000 images of 784 pixels corresponding to
handwritten digits form 0 to 9.  
The logistic regression, with 10 classes,  is tested with the 3 algorithms and some comments are provided, comparing the results.
Times are obtained by launching twice the example to avoid the compilation time of the first pass.
Run times are those obtained on a 4 hyperthreaded i7-cores laptop at 2.7Ghz

### SCSG logistic regression

The identifiability constraint was set on the class corresponding to the 0 digit contrary to the Rust tests
where the 9-digit class was chosen, this explains the different initial error and the fact that the best step
was not the same.
For the signification of the parameters B_0 , b_O, see documentation of SCSG.
Here we give some results:

- initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 8.88

| nb iter | B_0    |   m_0    | step_0  | y value | time(s) |
|  :---:  | :---:  |  :-----: | :----:  |   ----  |  ----   |
| 100     | 0.015  |  0.0015  |  0.5    |  0.58   |  35.    |
| 100     | 0.015  |  0.0015  |  0.25   |  0.33   |  34.    |
|  50     | 0.015  |  0.0015  |  0.25   |  0.42   |  17.8   |
|  50     | 0.015  |  0.0015  |  0.5    |  0.7    |  17.8   |

- initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter | B_0    |   m_0    | step_0  | y value  | time(s) |
|  ---    |----    |  ----    | ------  |   ----   |  ----  |
|  50     | 0.015  |  0.0015  |  0.25   |  0.39    |  18    |
|  100    | 0.015  |  0.0015  |  0.25   |  0.32    |  36    |

## Comparison with a Rust version

The Rust version has also an implementation of the SAG algorithm:

The Stochastic Averaged Gradient Descent as described in the paper:
**"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)**
M.Schmidt, N.LeRoux, F.Bach.

Here are some cpu-time comparisons :

Due to the different identifiability constraint the best result were not obtained with
the same step but we have the same comportment with respect to the initial condition
and equivalent results are obtained at about factor 1.5 in cpu-time. But
the Logistic Regression needed the explicit use of BLAS interface to speed up vectorization.

## License

Licensed under either of

- Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
- MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)
