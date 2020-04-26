
# MultiStochGrad

## This package provides an implementation of 2 stochastic gradient algorithms: SCSG and SVRG

 The algorithms implemented are described in the following papers:

* The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    1. "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    2. "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

* The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2013).  
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

These algorithms minimize functions given by an expression:  
**f(x) = 1/n ∑ fᵢ(x)** where fᵢ is a convex function.

All algorithms alternates some form of large batch computation (computing gradient of many terms of the sum)
and small or mini batches (computing a small number of terms, possibly just one, term of the gradient)
and updating position by combining these global and local gradients.
Further documentation can be found in docs of the Julia package, more in the doc of the Rust crate and at last the reference papers.

## Comparison with a Rust version

There is also a Rust version of this package.
The Rust version has also an implementation of the SAG algorithm:

The Stochastic Averaged Gradient Descent as described in the paper:
"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)
M.Schmidt, N.LeRoux, F.Bach.

Here are some cpu-time comparisons :

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)
