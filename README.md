
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2409.05518-b31b1b.svg)](https://arxiv.org/pdf/2409.05518v3)


# Description
This script implements a fixed-point iteration algorithm in JAX to solve a class of one-to-one matching models with linear transfers.

The main script, MatchingModel.py, sets up and solves a general system of fixed-point equations for the set of equilibrium transfers in a one-to-one matching model. The Script DiscreteChoiceModel.py contains the discrete choice models (Logit and Generalized Nested Logit). Finally, the script main.py sets up and solves three special cases of one-to-one matching models with transferable utility, where the discrete choices of agents on both sides of the matching market are described by:
 - Logit model
 - Nested logit model
 - Generalized nested logit model

# Citations
If you use this script in your research, I ask that you also cite [Andersen (2025)](https://arxiv.org/pdf/2409.05518v3).


    @article{andersen2025note,
      title={Note on solving one-to-one matching models with linear transferable utility}, 
      author={Esben Scrivers Andersen},
      year={2025},
      eprint={2409.05518},
      archivePrefix={arXiv},
      primaryClass={econ.GN},
      url={https://arxiv.org/pdf/2409.05518v3}, 
    }

# Dependencies
In order to run the scripts, you need to have installed [JAX](https://github.com/jax-ml/jax), [Simple Pytree](https://github.com/cgarciae/simple-pytree), and [FixedPointJAX](https://github.com/esbenscriver/FixedPointJAX). The latter is an implementation of a fixed-point iteration algorithm that allow the user to solve the system of fixed-point equations by standard fixed-point iterations and the SQUAREM accelerator, see [Du and Varadhan (2020)](https://www.jstatsoft.org/article/view/v092i07).

    




