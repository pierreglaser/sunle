# SMNLE

Vendoring of [SM-ExpFam-FLI](https://github.com/LoryPack/SM-ExpFam-LFI), the package implementing the SBI method
proposed in [Score Matched Neural Exponential Families for Likelihood-Free Inference](https://jmlr.org/papers/v23/21-0061.html).
The vendored version is a fork that
- includes a sbibm-type interface that performs end-to-end observation generation/training/inference (as opposed to the
  original code where these steps are separated into different python scripts) to easily test the method on other tasks
  than the one present in the paper.
- allows to use our custom doubly-intractable MCMC sampler, which automatically tunes important parameters.

Numerical reproducibility between an early version of the rewrite and the original code
(for the experiments of the original paper, and using their own sampler) is tested.
