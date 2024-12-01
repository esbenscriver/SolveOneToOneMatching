
[![arXiv](https://img.shields.io/badge/arXiv-2112.14377-b31b1b.svg)](https://arxiv.org/pdf/2409.05518)

# Description
This script implement in JAX a fixed point iteration algorithm to solve a class of one-to-one matching model with linear transfers.

The main script is SolveOneToOneMatching.py that sets up and solves a general system of fixed point equations for the set of equilibrium transfers in a one-to-one matching model. The script run_test.py uses the main script to solve three special cases of one-to-one matching models, where the discrete choices of the agents on both sides of the matching market are described by the
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
In order to run the scripts, you need to have installed JAX, Simple Pytree, and FixedPointJAX. The latter is an implementation of a fixed point iteration algorithm that allow the user to solve the system of fixed point equations by standard fixed point iterations and the SQUAREM accelerator, see [Du and Varadhan (2020)](https://www.jstatsoft.org/article/view/v092i07).

You can install the current release of [FixedPointJAX](https://github.com/esbenscriver/FixedPointJAX) with pip

    pip install FixedPointJAX
    
You can install the current release of [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) with pip

    pip install jax

You can install the current release of [Simple Pytree](https://github.com/cgarciae/simple-pytree) with pip

    pip install simple-pytree



