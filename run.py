# import JAX
import jax
import jax.numpy as jnp
from jax import random

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass, field, static_field

# import discrete-choice models
from ChoiceProbabilities import Logit, GNLogit

# import discrete-choice models
from SolveMatchingModel import EndogenousVariables

# Print numbers with 3 decimals
jnp.set_printoptions(precision=3)

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

# Specify the dimensions of the matching market and simulate structural
# parameters and distributions of agents
@dataclass
class SimulatedExogenousVariables(Pytree, mutable=True):
  acceleration: str = "None" # use standard fixed-point iterations
  # acceleration: str = "SQUAREM" # use accelerated fixed-point iterations

  def __init__(self, typesX=4, typesY=6, axisX=1, axisY=0, nestsX=2, nestsY=3):
    # Set number of unique types
    self.typesX = typesX
    self.typesY = typesY

    # Set axis describing the alternatives of workers (axisX) and firms (axisY)
    self.axisX = axisX
    self.axisY = axisY

    # Set number of nests describing the alternatives
    self.nestsX = nestsX
    self.nestsY = nestsY

    # Simulate matrices describing nesting structure of workers and firms
    self.nestingY = SimulateDummyMatrix(340, nestsX, typesX, axis=axisX)
    self.nestingX = SimulateDummyMatrix(333, nestsY, typesY, axis=axisY)

    # Simulate distribution of workers (nX) and firms(nY)
    self.nX: jnp.ndarray = random.uniform(key=random.PRNGKey(998),
                                          shape=(typesX, 1))
    self.nY: jnp.ndarray = random.uniform(key=random.PRNGKey(999),
                                          shape=(1, typesY)) + 1.0

    # Simulate preference parameters (βX) and productivity parameters (βY)
    self.utilityX: jnp.ndarray =-random.uniform(key=random.PRNGKey(111),
                                                shape=(typesX, typesY))
    self.utilityY: jnp.ndarray = random.uniform(key=random.PRNGKey(112),
                                                shape=(typesX, typesY))

    # Simulate scale parameters for workers (σX) and firms (σY)
    self.scaleX: jnp.ndarray = random.uniform(key=random.PRNGKey(113),
                                              shape=(typesX, 1)) + 1.0
    self.scaleY: jnp.ndarray = random.uniform(key=random.PRNGKey(114),
                                              shape=(1, typesY)) + 1.0

    # Simulate nesting parameters for workers (λX) and firms (λY)
    self.nestingParameterX: jnp.ndarray = random.uniform(key=random.PRNGKey(115),
                                                         shape=(typesX, 1, nestsY))
    self.nestingParameterY: jnp.ndarray = random.uniform(key=random.PRNGKey(116),
                                                         shape=(nestsX, 1, typesY))

    # Simulate nesting degree parameters for workers (αX) and firms (αY). The
    # logit transformation ensures that αX and αY sums to unit across nests
    self.nestingDegreeX: jnp.ndarray = Logit(random.uniform(key=random.PRNGKey(117),
                                                            shape=(1, typesY, nestsY)),
                                             axis=2, 
                                             outside=False)
    self.nestingDegreeY: jnp.ndarray = Logit(random.uniform(key=random.PRNGKey(118),
                                                           shape=(nestsX, typesX, 1)),
                                            axis=0, 
                                            outside=False)

# Simulate exogenous varialbes of the model
exog = SimulatedExogenousVariables()

print('-----------------------------------------------------------------------')
print(f'number of unique types of workers: {exog.typesX}')
print(f'number of unique types of firms: {exog.typesY}')
print(f'axis describing the choice set for workers: {exog.axisX}')
print(f'axis describing the choice set for firms: {exog.axisY}')
print(f'number of nests of firms: {exog.nestsX}')
print(f'number of nests of workers: {exog.nestsY}')
print('-----------------------------------------------------------------------')
print(f"{exog.utilityX.shape = }")
print(f"{exog.utilityY.shape = }")
print(f"{exog.scaleX.shape = }")
print(f"{exog.scaleY.shape = }")
print(f"{exog.nestingX.shape = }")
print(f"{exog.nestingY.shape = }")
print(f"{exog.nestingParameterX.shape = }")
print(f"{exog.nestingParameterY.shape = }")
print(f"{exog.nestingDegreeX.shape = }")
print(f"{exog.nestingDegreeY.shape = }")

# Set scalars for dampening the step length for the logit model
cX_logit = 1.0
cY_logit = 1.0

# Set up functions for logit choice probabilities as functions of scaled
# payoffs for workers (vX) and firms (vY)
probX_logit = lambda vX: Logit(vX, axis=exog.axisX)
probY_logit = lambda vY: Logit(vY, axis=exog.axisY)

# Find the endogenous variables: equilibrium distribution of wages and matches
print('-----------------------------------------------------------------------')
print('Solving the logit matching model:')
endog_logit = EndogenousVariables(cX_logit,
                                  cY_logit,
                                  probX_logit,
                                  probY_logit,
                                  exog)

# Set scalars for dampening the step length for the nested logit model
cX_nested_logit = jnp.sum(exog.nestingX * exog.nestingParameterX, axis=2)
cY_nested_logit = jnp.sum(exog.nestingY * exog.nestingParameterY, axis=0)

# Set up functions for nested logit choice probabilities as functions of scaled
# payoffs for workers (vX) and firms (vY)
probX_nested_logit = lambda vX: GNLogit(vX, 
                                        degree=exog.nestingX, 
                                        nesting=exog.nestingParameterX, 
                                        axis=exog.axisX)

probY_nested_logit = lambda vY: GNLogit(vY, 
                                        degree=exog.nestingY, 
                                        nesting=exog.nestingParameterY, 
                                        axis=exog.axisY)

print('-----------------------------------------------------------------------')
print('Solving the nested logit matching model:')
endog_nested_logit = EndogenousVariables(cX_nested_logit,
                                         cY_nested_logit,
                                         probX_nested_logit,
                                         probY_nested_logit,
                                         exog)

# Set scalars for dampening the step length for the nested logit model
cX_GNLogit = jnp.min(jnp.squeeze(exog.nestingParameterX), axis=exog.axisX, keepdims=True)
cY_GNLogit = jnp.min(jnp.squeeze(exog.nestingParameterY), axis=exog.axisY, keepdims=True)

# Set up functions for nested logit choice probabilities as functions of scaled
# payoffs for workers (vX) and firms (vY)
probX_GNLogit = lambda vX: GNLogit(vX, 
                                   degree=exog.nestingDegreeX, 
                                   nesting=exog.nestingParameterX, 
                                   axis=exog.axisX)

probY_GNLogit = lambda vY: GNLogit(vY, 
                                   degree=exog.nestingDegreeY, 
                                   nesting=exog.nestingParameterY, 
                                   axis=exog.axisY)

print('-----------------------------------------------------------------------')
print('Solving the generalized nested logit matching model:')
endog_GNLogit = EndogenousVariables(cX_GNLogit,
                                    cY_GNLogit,
                                    probX_GNLogit,
                                    probY_GNLogit,
                                    exog)