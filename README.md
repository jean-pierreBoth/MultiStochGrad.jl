
# MultiStochGrad.jl

 **This package provides an interface to the rust crate multistochgrad that implments
 3 stochastic gradient algorithms**

 The algorithms described in the following papers:

* The so-called SCSG algorithm described and analyzed in the two papers by L. Lei and  M.I Jordan.

    1. "On the adaptativity of stochastic gradient based optimization" (2019)
    [SCSG-1](https://arxiv.org/abs/1904.04480)

    2. "Less than a single pass : stochastically controlled stochastic gradient" (2019)
    [SCSD-2](https://arxiv.org/abs/1609.03261)

* The SVRG algorithm described in the paper by R. Johnson and T. Zhang
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction" (2013).  
[Advances in Neural Information Processing Systems, pages 315–323, 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)

## Rust installation and crate hnsw-rs installation

* Rust installation see [Rust Install](https://www.rust-lang.org/tools/install)

run :  
curl https://sh.rustup.rs -sSf | sh

   The hnsw_rs package can be downloaded from [Hnsw](https://gitlab.com/jpboth/hnswlib-rs) or soon
   from [crate.io](https://crates.io/).

* compilation of rust library.
    By default the rust crate builds a static library. The ***Building*** paragraph in the README.md file of the rust crate, describes how to build the dynamic libray needed for use with Julia.

* Then push the path to the library *libhnsw_rs.so* in Base.DL_LOAD_PATH
(see this package function setRustlibPath(path::String)

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)
