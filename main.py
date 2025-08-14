"""
Solve three examples of one-to-one matching model with linear transfers, where the discrete choices of the agents 
on both sides of the matching market are described by:
 - Logit model
 - Nested logit model
 - Generalized nested logit model

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from module.MatchingModel import MatchingModel
from module.DiscreteChoiceModel import LogitModel, NestedLogitModel, GeneralizedNestedLogitModel

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

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

# Set nesting index (used for nested logit model)
nest_index_X = jnp.arange(types_Y) % nests_Y
nest_index_Y = jnp.arange(types_X) % nests_X

# Simulate nesting parameters (used for nested logit and generalized nested logit model)
nest_parameter_X = random.uniform(key=random.PRNGKey(211), shape=(types_X, nests_Y), minval=0.1, maxval=1.0)
nest_parameter_Y = random.uniform(key=random.PRNGKey(212), shape=(types_Y, nests_X), minval=0.1, maxval=1.0)

# Simulate nesting structure of the generalized nested logit model
nest_share_X = random.dirichlet(key=random.PRNGKey(311), alpha=jnp.ones((nests_Y,)), shape=(types_Y,))
nest_share_Y = random.dirichlet(key=random.PRNGKey(312), alpha=jnp.ones((nests_X,)), shape=(types_X,))

print('-----------------------------------------------------------------------')
print('1. Solve a matching model with logit demand:')

model_logit = MatchingModel(
  model_X = LogitModel(utility=utility_X, scale=scale_X, n=n_X),
  model_Y = LogitModel(utility=utility_Y, scale=scale_Y, n=n_Y),  
)

solution_logit = model_logit.Solve(acceleration=acceleration)

print('-----------------------------------------------------------------------')
print('2. Solve a matching model with nested logit demand:')

model_nested_logit = MatchingModel(
  model_X = NestedLogitModel(
    utility=utility_X, 
    scale=scale_X,

    nest_index=nest_index_X,
    nest_parameter=nest_parameter_X,

    n=n_X,
  ),

  model_Y = NestedLogitModel(
    utility=utility_Y, 
    scale=scale_Y,

    nest_index=nest_index_Y,
    nest_parameter=nest_parameter_Y,

    n=n_Y,
  ),
)

solution_nested_logit = model_nested_logit.Solve(acceleration=acceleration)

print('-----------------------------------------------------------------------')
print('3. Solve a matching model with generalized nested logit demand:')

model_GNL = MatchingModel(
  model_X=GeneralizedNestedLogitModel(
    utility=utility_X, 
    scale=scale_X,

    nest_share=nest_share_X,
    nest_parameter=nest_parameter_X,

    n=n_X,
  ),

  model_Y = GeneralizedNestedLogitModel(
    utility=utility_Y, 
    scale=scale_Y,

    nest_share=nest_share_Y,
    nest_parameter=nest_parameter_Y,

    n=n_Y,
  ),
)

solution_GNL = model_GNL.Solve(acceleration=acceleration)