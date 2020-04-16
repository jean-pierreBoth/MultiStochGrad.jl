
# MultiStochGrad.jl

 **This package provides an interface to the rust crate multistochgrad that implements
 3 stochastic gradient algorithms**

 The algorithms implemented are described in the following papers:

* The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    1. "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    2. "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

* The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2013).  
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

* The SAG algorithm described in :

The Stochastic Averaged Gradient Descent as described in the paper:
"Minimizing Finite Sums with the Stochastic Average Gradient" (2013, 2016)
M.Schmidt, N.LeRoux, F.Bach

All algorithms alternates some form of large batch computation (computing gradient of many terms of the sum)
and small or mini batches (computing a small number of terms, possibly just one, term of the gradient)
and updating position by combining these global and local gradients.
Further documentation can be found in docs of the Julia package, more in the doc of the Rust crate and at last the reference papers.

## Rust installation and crate multistochgrad installation

* Rust installation see [Rust Install](https://www.rust-lang.org/tools/install)

run :  
curl https://sh.rustup.rs -sSf | sh

   The multistochgrad package can be downloaded from [Hnsw](https://github.com/jean-pierreBoth/multistochgrad) or soon
   from [crate.io](https://crates.io/).

* compilation of rust library.
    By default the rust crate builds a static library. The ***Building*** paragraph in the README.md file of the rust crate, describes how to build the dynamic libray needed for use with Julia.

* Then push the path to the library *libmultistochgrad_rs.so* in Base.DL_LOAD_PATH
(see this package function setRustlibPath(path::String)

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)
