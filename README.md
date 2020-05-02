
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

## Structure of the package and documentation

The structure of the package is explicited in the documentation along with a html page for each implemented SGD.  

Doc is generated as usual by running julia make.jl in the docs/src directory.  

Further documentation can be found in docs of the Julia package, more in the doc of the Rust crate and at last the reference papers.

## Examples and tests

Small tests consist in a line fitting problem that is taken  from the crate optimisation.

Examples are based on logisitc regression applied to digits MNIST database
(as in the second paper on SCSG).  
The data files can be downloaded from [MNIST](http://yann.lecun.com/exdb/mnist).
The database has 60000 images of 784 pixels corresponding to
handwritten digits form 0 to 9.  
The logistic regression, with 10 classes,  is tested with the 2 algorithms and some comments are provided, comparing the results.
Times are obtained by launching twice the example to avoid the compilation time of the first pass.
Run times are those obtained on a 4 hyperthreaded i7-cores laptop at 2.7Ghz

### SCSG logistic regression

The identifiability constraint was set on the class corresponding to the 0 digit. (Contrary to the Rust tests
where the 9-digit class was chosen, this explains the different initial error and the fact that the best step
was not the same).

For the signification of the parameters B_0 , b_O, see documentation of SCSG.
Here we give some results:

- initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 8.88

| nb iter | B_0    |   m_0    | step_0  | y value | time(s) |
|  :---:  | :---:  |  :-----: | :----:  |   ----  |  ----   |
|  50     | 0.015  |  0.003   |  0.1    |  0.296  |  11.8   |
|  50     | 0.02   |  0.004   |  0.1    |  0.29   |  13.3   |
|  50     | 0.02   |  0.006   |  0.1    |  0.28   |  16.5   |
|  70     | 0.02   |  0.006   |  0.1    |  0.27   |  23.8   |

- initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter | B_0    |   m_0    | step_0  | y value  | time(s) |
|  ---    | :----: |  ----    | ------  |   ----   |  ----  |
|  50     | 0.015  |  0.003   |  0.1    |  0.286   |  12    |
|  50     | 0.02   |  0.004   |  0.1    |  0.28    |  13.5  |
|  50     | 0.02   |  0.006   |  0.1    |  0.276   |  16.5  |
|  100    | 0.02   |  0.004   |  0.1    |  0.266   |  26    |

### SVRG logistic regression

- initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 2.3

| nb iter |  nb mini batch     | step    | y value  | time(s) |
|  ---    |     :---:          | ------  |   ----   |  ----   |
|  50     |     1000           |  0.05   |  0.269   |  28     |  
|  50     |     2000           |  0.05   |  0.255   |  29     |  

- initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter |  nb mini batch     | step    | y value  | time(s) |
|  ---    |     :---:          | ------  |   ----   |  ----  |
|  50     |     500            |  0.05   |  0.28    |  23.5  |
|  50     |     1000           |  0.05   |  0.264   |  24.5  |  
|  50     |     2000           |  0.05   |  0.25    |  27    |  
|  100     |    1000           |  0.05   |  0.253   |  50    |


We see that SCSG is fast to reach a minimum of 0.28, it is more difficult to reach 0.26-0.27
it nevertheless faster than SVRG.  
The efficiency gain with respect to SVRG is not as important
as with the *Rust* version (where we have a factor 2 in cpu-time) *probably* beccause our test use 60000
observations, with SCSG we run at most on 6000 terms in a batch so, whick is not enough to compensate
the overhead of threading. It should be more efficient in larger tests.
Nevertheless the Julia version within a factor 1.5 or 2 of the Rust version.

## Rust version of this package

There is also a Rust implementation of this package at [multigrad.rs](https://github.com/jean-pierreBoth/multistochgrad).  

The Rust version has also an implementation of the SAG algorithm:

The Stochastic Averaged Gradient Descent as described in the paper:
**"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)**
M.Schmidt, N.LeRoux, F.Bach.

### cpu-time comparisons  

Due to the different identifiability constraint the best result were not obtained with
the same step size in gradient but we have the same behaviour with respect to the initial condition
and equivalent results are obtained within a factor 2 in cpu-time. But
the logistic regression needed the explicit use of BLAS interface to speed up vectorization.

## License

Licensed under either of

- Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
- MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)
