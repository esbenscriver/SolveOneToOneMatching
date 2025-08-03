"""
Solve three examples of one-to-one matching model with linear transfers, where the discrete choices of the agents 
on both sides of the matching market are described by:
 - Logit model
 - Nested logit model
 - Generalized nested logit model

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518v3)
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

# import solver for one-to-one matching model
from module.MatchingModel import MatchingModel
from module.DiscreteChoiceModel import LogitModel, NestedLogitModel, GeneralizedNestedLogitModel

# choose accelerator of fixed-point iterations
acceleration = "None"
# acceleration = "SQUAREM"

# Set number of types of agents on both sides of the market
types_X, types_Y = 4, 6

# Simulate choice-specific utilities
utility_X =-random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
utility_Y = random.uniform(key=random.PRNGKey(112), shape=(types_Y, types_X))

# Simulate scale parameters
scale_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1)) + 1.0
scale_Y = random.uniform(key=random.PRNGKey(114), shape=(types_Y, 1)) + 1.0

# Simulate distribution of agents
n_X = random.uniform(key=random.PRNGKey(115), shape=(types_X, 1))
n_Y = random.uniform(key=random.PRNGKey(116), shape=(types_Y, 1)) + 1.0

# Set number of nests
nests_X, nests_Y = 2, 3

# Simulate nesting parameters
nesting_parameter_X = random.uniform(key=random.PRNGKey(211), shape=(types_X, nests_Y), minval=0.1, maxval=1.0)
nesting_parameter_Y = random.uniform(key=random.PRNGKey(212), shape=(types_Y, nests_X), minval=0.1, maxval=1.0)

print('-----------------------------------------------------------------------')
print('1. Solve a matching model with logit demand:')

model_logit = MatchingModel(
  model_X = LogitModel(utility=utility_X, scale=scale_X, n=n_X),
  model_Y = LogitModel(utility=utility_Y, scale=scale_Y, n=n_Y),  
)

model_logit.Solve(acceleration=acceleration)

print('-----------------------------------------------------------------------')
print('2. Solve a matching model with nested logit demand:')

model_nested_logit = MatchingModel(
  model_X = NestedLogitModel(
    utility=utility_X, 
    scale=scale_X,

    nesting_index=jnp.arange(types_Y) % nests_Y,
    nesting_parameter=nesting_parameter_X,

    n=n_X,
  ),

  model_Y = NestedLogitModel(
    utility=utility_Y, 
    scale=scale_Y,

    nesting_index=jnp.arange(types_X) % nests_X,
    nesting_parameter=nesting_parameter_Y,

    n=n_Y,
  ),
)

model_nested_logit.Solve(acceleration=acceleration)

print('-----------------------------------------------------------------------')
print('3. Solve a matching model with generalized nested logit demand:')

model_GNL = MatchingModel(
  model_X=GeneralizedNestedLogitModel(
    utility=utility_X, 
    scale=scale_X,

    nesting_structure=random.dirichlet(key=random.PRNGKey(311), alpha=jnp.ones((nests_Y,)), shape=(types_Y,)),
    nesting_parameter=nesting_parameter_X,

    n=n_X,
  ),

  model_Y = GeneralizedNestedLogitModel(
    utility=utility_Y, 
    scale=scale_Y,

    nesting_structure=random.dirichlet(key=random.PRNGKey(312), alpha=jnp.ones((nests_X,)), shape=(types_X,)),
    nesting_parameter=nesting_parameter_Y,

    n=n_Y,
  ),
)

model_GNL.Solve(acceleration=acceleration)