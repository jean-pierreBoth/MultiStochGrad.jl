
# MultiStochGrad

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

The user interacts mainly with two structures defining the optimization problem (detailed in file **evaluator.jl**) specifying how to compute each term value and gradient:

    - struct TermFunction{F <: Function}

    - struct TermGradient{G <: Function>}
  
The structure of the package is explicited in the documentation along with a html page for each implemented SGD.  

Doc is generated as usual by running julia make.jl in the docs/src directory.  

Further documentation can be found in docs of the Julia package, more in the doc of the Rust crate and at last the reference papers.

## Examples and tests

Small tests consist in a line fitting problem and logisitc regression.

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
|  50     | 0.02   |  0.004   |  0.1    |  0.29   |  11.3   |
|  50     | 0.02   |  0.006   |  0.1    |  0.281  |  14.    |
|  70     | 0.02   |  0.006   |  0.1    |  0.272  |  19.5   |
|  100    | 0.02   |  0.006   |  0.1    |  0.265  |  27.    |

- initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter | B_0    |   m_0    | step_0  | y value  | time(s) |
|  ---    | :----: |  ----    | ------  |   ----   |  ----  |
|  50     | 0.015  |  0.003   |  0.1    |  0.285   |  10.2  |
|  50     | 0.02   |  0.004   |  0.1    |  0.28    |  11    |
|  50     | 0.02   |  0.006   |  0.1    |  0.274   |  14    |
|  100    | 0.02   |  0.004   |  0.1    |  0.266   |  22.   |
|  100    | 0.02   |  0.006   |  0.1    |  0.262   |  27.5  |

### SVRG logistic regression

- initialization position : 9 images with *constant pixel = 0.5*,
error at initial position: 8.88

| nb iter |  nb mini batch     | step    | y value  | time(s) |
|  ---    |     :---:          | ------  |   ----   |  ----   |
|  50     |     1000           |  0.05   |  0.269   |  25     |  
|  50     |     2000           |  0.05   |  0.255   |  27     |  

- initialization position : 9 images with *constant pixel = 0.0*,
error at initial position: 2.3

| nb iter |  nb mini batch     | step    | y value  | time(s) |
|  ---    |     :---:          | ------  |   ----   |  ----  |
|  50     |     500            |  0.05   |  0.28    |  23.5  |
|  50     |     1000           |  0.05   |  0.264   |  24.5  |  
|  50     |     2000           |  0.05   |  0.25    |  27    |  
|  100     |    1000           |  0.05   |  0.253   |  50    |

### some comments on cpu-times

All times were obtained withe @time macro and julia running with JULIA_NUM_THREADS = 8.

We see that SCSG is fast to reach a minimum of 0.28, it is more difficult to reach 0.26-0.27
it nevertheless quite competitive compared to SVRG.  
The efficiency gain with respect to SVRG is not as important
as with the *Rust* version (see below) where we have a factor 1.8 in cpu-time **due to a multithreading effect**.
Our test uses 60000 observations and SCSG runs at most on 1/10 of the terms in a batch (i.e 6000 terms), on the constrary SVRG batches run on the full 60000 terms.  
**Batch sizes on SCSG are not large enough to fully benefit of the multithreading.**
This can be verified by setting JULIA_NUM_THREADS = 1 and compare how SCSG and SVRG timing benefit from
the threading:

We did 2  checks with initial conditions set to pixel = 0.5 and compared with previous results:

- For SCSG we ran the case with parameters (70, 0.02, 0.006, 0.1 )  corresponding to last line of first array of results. We had with 8 threads y=0.27 in 20s , and with one thread we obtain y=0.27 in 36.5s, so the threading gives us less than a factor 2.

- For SVRG we ran the case with parameters (50,1000, 0.05) corresponding to the first line of first array of result for SVRG.
We had y=0.269 with 25s, and with one thread we need 72s. Here the threading
gives us a gain of 3.

So SCSG  efficiency should be more evident on larger problems, note that the threading in Julia is still young.

The logistic regression needed the explicit use of BLAS interface to speed up vectorization.

*Nevertheless the Julia version runs within a factor 1.5 or 1.8 of the Rust version which seems a good compromise.*

## Rust version of this package

There is also a Rust implementation of this package at [multigrad.rs](https://github.com/jean-pierreBoth/multistochgrad).  

The Rust version has also an implementation of the SAG algorithm:

The Stochastic Averaged Gradient Descent as described in the paper:
**"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)**
M.Schmidt, N.LeRoux, F.Bach.

## Note

- version 0.1.2 runs 10-15 % faster on SCSG due to some optimization
see  [The need for rand speed](https://bkamins.github.io/julialang/2020/11/20/rand.html) or julia bloggers.

## License

Licensed under either of

- Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
- MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)
