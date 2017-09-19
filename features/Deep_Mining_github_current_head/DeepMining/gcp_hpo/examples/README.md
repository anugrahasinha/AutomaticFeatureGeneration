# Examples

The `gcp_hpo.examples` module provides some use cases of the SmartSearch class for hyper-parameter optimization.

`interface` shows two different ways of using SmartSearch with basic examples: with a custom function that one 
wants to optimize, or with an sklearn pipeline.

`branin` and `har6` can be used to see how randomized/GP-based/GCP-based search perform on these two functions, 
that are often used in optimization.