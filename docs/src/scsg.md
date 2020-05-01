# SCSG

## Description

The notations are those of the paper:  

**On the adaptativity of stochastic gradient based optimization" (2019).**

One iteration j consists in :

- a large batch of size Bⱼ
- a number noted mⱼ of small batches of size bⱼ
- update position with a step ηⱼ.

The number of mini batch is descrbed by a random variable with a geometric law.  
The papers establishes rates of convergence depending on the ratio
mⱼ/Bⱼ , bⱼ/mⱼ and ηⱼ/bⱼ and their products.

 The second paper :  
 **"Less than a single pass : stochastically controlled stochastic gradient"**  

describes a simplified version where the mini batches consist in just one term
and the number of mini batch is set to the mean of the geometric variable corresponding to
number of mini batches.  

We adopt a mix of the two papers:  
It seems letting the size of mini batch grow a little is more stable than keeping it to 1
(in particular when initialization of the algorithm varies).  
Also replacing the geometric law by its mean is really more stable due to the large variance of its law
as recommended in the second paper.  

With the following notations:

- nbterms : the number of terms in the objective function
- nbiter : the maximum number of iterations
- j : the iteration number

We choose to let Bⱼ, mⱼ, bⱼ and ηⱼ evolve as: 

       Bⱼ  :   B₀ * nbterms * alfa^(2j)
       mⱼ  :   m₀ * alfa^(3j/2)
       bⱼ  :   b₀ * alfa^j
       ηⱼ  :   η₀ / alfa^(j/2)
     
     where alfa is computed to be slightly greater than 1.  
     α is chosen so that :   B₀ * α^(2*nbiter) = 1.

The evolution of Bⱼ is bounded above by nbterms/10 and bⱼ by nbterms/100.  
The size of small batch must stay small so b₀ must be small (typically 1 seems OK)
