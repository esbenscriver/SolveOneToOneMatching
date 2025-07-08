"""
Solve three examples of one-to-one matching model with linear transfers. 

The discrete choices of the agents on both sides of the matching market are described by the
 - Logit model
 - Nested logit model
 - Generalized nested logit model

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2024 (https://arxiv.org/pdf/2409.05518)
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

# import solver for one-to-one matching model
from SolveOneToOneMatching import Logit, GNLogit, ExogenousVariables, MatchingModel

def SimulateDummyMatrix(key: int, nests: jnp.ndarray, types: int, axis: int):
  """Simulate matrix describing the nesting structure of the nested logit model
  for the agents on the other side of the market.

      Inputs:
        - key:
        - types: number of types
        - nests: number of nests
        - axis: describe the dimension of the agents of the other side

      Outputs:
        - matrix describing the nesting structure
  """
  # ID of the nest
  nestID = jnp.expand_dims(jnp.arange(nests), axis=axis)

  # simulate which nest the alternatives belong to
  alternativeNestID = jnp.expand_dims(random.randint(key=random.PRNGKey(key),
                                                     minval=0,
                                                     maxval=nests,
                                                     shape=(types,)),
                                      axis=1-axis)

  return jnp.expand_dims(jnp.where(nestID == alternativeNestID, 1.0, 0.0), axis=-axis)

# choose accelerator
acceleration = "None"
# acceleration = "SQUAREM"

# Set size of matching market
typesX, typesY = 4, 6

# Set number of nests
nestsX, nestsY = 2, 3

# simulate exogenous variables of the matching model
exog = ExogenousVariables(
  axisX = 1, # set axis that describe the alternatives in the workers' choice set
  axisY = 0, # set axis that describe the alternatives in the firms' choice set

  # Simulate choice-specific utilities
  utilityX =-random.uniform(key=random.PRNGKey(111), shape=(typesX, typesY)),
  utilityY = random.uniform(key=random.PRNGKey(112), shape=(typesX, typesY)),

  # Simulate scale parameters
  scaleX = random.uniform(key=random.PRNGKey(113), shape=(typesX, 1)) + 1.0,
  scaleY = random.uniform(key=random.PRNGKey(114), shape=(1, typesY)) + 1.0,

  # Simulate distribution of workers and firms
  nX = random.uniform(key=random.PRNGKey(998), shape=(typesX, 1)),
  nY = random.uniform(key=random.PRNGKey(999), shape=(1, typesY)) + 1.0,
)

# Simulate nesting parameters for workers (λX) and firms (λY)
nestingParameterX = random.uniform(key=random.PRNGKey(115), shape=(typesX, 1, nestsY))
nestingParameterY = random.uniform(key=random.PRNGKey(116), shape=(nestsX, 1, typesY))

# Simulate matrices describing nesting structure of workers and firms
nestingY = SimulateDummyMatrix(340, nestsX, typesX, axis=exog.axisX)
nestingX = SimulateDummyMatrix(333, nestsY, typesY, axis=exog.axisY)

# Simulate nesting degree parameters for workers (αX) and firms (αY).
nestingDegreeX= Logit(
  random.uniform(key=random.PRNGKey(117), shape=(1, typesY, nestsY)),
  axis=2, 
  outside_option=False,
)
nestingDegreeY = Logit(
  random.uniform(key=random.PRNGKey(118), shape=(nestsX, typesX, 1)),
  axis=0, 
  outside_option=False,
)

# Find the endogenous variables: equilibrium distribution of wages and matches
print('-----------------------------------------------------------------------')
print('Solving the logit matching model:')
model_logit = MatchingModel(
  exog = exog,

  # Set up logit choice probabilities
  prob_X = lambda vX: Logit(vX, axis=exog.axisX, outside_option=True),
  prob_Y = lambda vY: Logit(vY, axis=exog.axisY, outside_option=True),

  # Set scalars used for adjusting step length of fixed-point iteration
  cX = 1.0,
  cY = 1.0,
)

model_logit.Solve(acceleration=acceleration)

print('-----------------------------------------------------------------------')
print('Solving the nested logit matching model:')
model_nested_logit = MatchingModel(
  exog = exog,

  # Set up nested logit choice probabilities
  prob_X = lambda vX: GNLogit(
    vX, 
    degree=nestingX, 
    nesting=nestingParameterX, 
    axis=exog.axisX
  ),
  prob_Y = lambda vY: GNLogit(
    vY, 
    degree=nestingY, 
    nesting=nestingParameterY, 
    axis=exog.axisY,
  ),

  # Set scalars used for adjusting step length of fixed-point iteration
  cX = jnp.sum(nestingX * nestingParameterX, axis=2),
  cY = jnp.sum(nestingY * nestingParameterY, axis=0),
)

model_nested_logit.Solve(acceleration=acceleration)

print('-----------------------------------------------------------------------')
print('Solving the generalized nested logit matching model:')
model_GNL = MatchingModel(
  exog = exog,

  # Set up generalized nested logit choice probabilities
  prob_X = lambda vX: GNLogit(
    vX, 
    degree=nestingDegreeX, 
    nesting=nestingParameterX, 
    axis=exog.axisX,
  ),

  prob_Y = lambda vY: GNLogit(
    vY, 
    degree=nestingDegreeY, 
    nesting=nestingParameterY, 
    axis=exog.axisY,
  ),

  # Set scalars used for adjusting step length of fixed-point iteration
  cX = jnp.min(jnp.squeeze(nestingParameterX), axis=exog.axisX, keepdims=True),
  cY = jnp.min(jnp.squeeze(nestingParameterY), axis=exog.axisY, keepdims=True),
)

model_GNL.Solve(acceleration=acceleration)