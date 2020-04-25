# SCSG

## Description

One iteration j consists in a large batch of size Bⱼ and then a number noted mⱼ of small batches
of size bⱼ and update position with a step ηⱼ.
The number of mini batch is descrbed by a random variable with a geometric law.  
The papers establishes rates of convergence depending on the ratio
mⱼ/Bⱼ , bⱼ/mⱼ and ηⱼ/bⱼ and their products.  
We stick to the description made in this paper. The second paper :  
 **"Less than a single pass : stochastically controlled stochastic gradient"**  
describes a simplified version where the mini batches consist in just one term
and the number of mini batch is set to the mean of the geometric variable corresponding to
number of mini batches.  

We adopt a mix of the two papers:  
It seems letting the size of mini batch grow a little is more stable than keeping it to 1.
(in particular when initialization of the algorithm varies.)
but replacing the geometric law by its mean is really more stable due to the large variance of its law.

If nbterms is the number of terms in function to minimize and j the iteration number:

       Bⱼ evolves as :   large_batch_size_init * nbterms * alfa^(2j)
       mⱼ evolves as :   m_zero * alfa^(3j/2)
       bⱼ evolves as :   b_0 * alfa^j
       ηⱼ evolves as :   eta_0 / alfa^j
     
     where alfa is computed to be slightly greater than 1.  
    In fact α is chosen so that :  B_0 * alfa^(2*nbiter) = nbterms

The evolution of Bⱼ is bounded above by nbterms/10 and bⱼ by nbterms/100.  
The size of small batch must stay small so b₀ must be small (typically 1 seems OK)
