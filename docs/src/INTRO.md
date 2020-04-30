# Introduction

These algorithms minimize functions given by an expression:

```math
f(x) = \frac{1}{n} \sum_{i=1}^{i=n} f_{i}(x)
```

where ``f_{i}`` is a convex function (possibly after regularization, as adding a regularization term
to f is just equivalent to adding it to each ``f_{i}``).

The variable ``x`` is a parameter to identify and each ``f_{i}`` is a loss function related to an observation,
so the number of terms in the sum is the number of observations.

## Structure of the package

The various stochastic gradient algorithms rely on different strategies alternating computations of gradients
of a large number of terms of the sum and a sequence of small number gradients computations to spare cpu times.

The package is structured by separating the strategy of a specific gradient algorithm (ex **struct SCSG**) and the evaluation  of the loss function **f** and its gradient which are respectively stored in the structures **struct TermFunction{F}** and **struct TermGradient{G}**.  
These structures store user provided functions
corresponding to the minimization problem to solve and observations of the problem. They can thus compute values and gradients of each ``f_{i}``.  
These 2 structures are stored in another **struct Evaluator{F,G}**.  

The various functions associated to **struct Evaluator{F,G}** provide parallelization of the calls to user provided functions for evaluation of values and gradients on many terms.

The function **minimize** runs the various gradient algorithms by being provided the arguments
describing the algorithm to run and the evaluations functions.