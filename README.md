A Discrete Formulation of Neural ODE
====================================


The mathematics of the original Neural ODE paper is densely presented
and it might present a challenge to readers of programming background.
In Figure 2 of the original paper, it is mentioned that \"if the loss
depends directly on the state at multi- ple observation times, the
adjoint state must be updated in the direction of the partial derivative
of the loss with respect to each observation\". However, Algorithm 1
only contains a single observation $z(t_1)$ and it is not obvious how
such updates should be done.

This article presents a discrete formulation of Section 2 of the
original Neural ODE paper. We intentionally broken down mathematical
formuation so it is easier to follow and correspondance to programming
implementation is explicitly provided.

