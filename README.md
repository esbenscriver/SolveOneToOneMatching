# Scripts for solving one-to-one matching models
This script implement in JAX a fixed point iteration algorithm to solve a class of one-to-one matching model with linear transfers.

The main script is SolveOneToOneMatching.py that sets up and solve a general system of fixed point equations for equilibrium transfers in a one-to-one matching model. The script run_test.py uses the main script to solve three special cases of a one-to-one matching model, where the discrete choices of the agents on both sides of the matching market are described by the
 - Logit model
 - Nested logit model
 - Generalized nested logit model

# Citations
If you use this script in your research, I ask that you also cite [Andersen (2024)](https://arxiv.org/pdf/2409.05518).


    @article{andersen2024notesolvingonetoonematching,
      title={Note on solving one-to-one matching models with linear transferable utility}, 
      author={Esben Scrivers Andersen},
      year={2024},
      eprint={2409.05518},
      archivePrefix={arXiv},
      primaryClass={econ.GN},
      url={https://arxiv.org/abs/2409.05518}, 
    }

# Installation
In order to run the scripts, you need to have installed JAX and my implementation in JAX of a fixed point iteration algorithm called FixedPointJAX. This implementation allow the user to solve the system of fixed point equations by standard fixed point iterations and the SQUAREM accelerator, see [Du and Varadhan (2020)](https://www.jstatsoft.org/article/view/v092i07).

You can install the current release of FixedPointJAX with [pip](https://pypi.org/project/FixedPointJAX/)

    pip install FixedPointJAX
