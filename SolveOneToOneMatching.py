import jax.numpy as jnp

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass, field, static_field

# import fixed-point iterator
from FixedPointJAX import FixedPointRoot

# JAX code to solve a one-to-one matching model with transferable utility

def Logit(v: jnp.ndarray, axis: int = 0, outside: bool = True) -> jnp.ndarray:
    """Calculates the logit choice probabilitie of the inside options

        Inputs:
         - v: match-specific payoffs
         - axis: tells which dimensions to sum over

        Outputs:
         - choice probabilities of matching with any type.
    """
    # exponentiated payoff of outside option
    expV0 = jnp.where(outside, 1.0, 0.0)

    # exponentiated payoffs of inside options (nominator)
    nominator = jnp.exp(v)

    # denominator of choice probabilities
    denominator = expV0 + jnp.sum(nominator, axis=axis, keepdims=True)
    return nominator / denominator

def GNLogit(v: jnp.ndarray, 
            degree: jnp.ndarray, 
            nesting: jnp.ndarray, 
            axis: int = 0, 
            outside: bool = True) -> jnp.ndarray:
    """Calculates the generalized nested logit choice probabilitie of the inside
    options

        Inputs:
         - v: match-specific payoffs
         - degree: degree the alternatives belong to the different nests
         - nesting: nesting parameters
         - axis: tells which dimensions to sum over

        Outputs:
         - choice probabilities of matching with any type.
    """
    # set dimensions of nests, alternatives, and types
    axisN = 2 * axis # axis for nests
    axisA = 1 # axis for alternatives

    # Set up functions for calculating choice probabilities
    expV = degree * jnp.exp(jnp.expand_dims(v, axis=axisN) / nesting)
    expV0 = jnp.where(outside, 1.0, 0.0)
    expV_nest = jnp.sum(expV, axis=axisA, keepdims=True)

    nominator = jnp.sum(expV * (expV_nest ** (nesting - 1.0)), axis=axisN)
    denominator = expV0 + jnp.sum(jnp.squeeze(expV_nest ** nesting),
                                  axis=axis,
                                  keepdims=True)
    return nominator / denominator

def UpdateTransfer(t_init: jnp.ndarray, 
                   K: jnp.ndarray, 
                   nX: jnp.ndarray, 
                   nY: jnp.ndarray, 
                   probX: callable, 
                   probY: callable) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates excess demand and updates fixed point equation for transfers

        Inputs:
         - t_init: array containing initial transfers
         - K: matrix describing the adjustment of the step lenght
         - nX: matrix containing the distribution of workers
         - nY: matrix containing the distribution of firms
         - probX: workers' choice probabilities as function of transfers
         - probY: firms' choice probabilities as function of transfers

        Outputs:
         - t_update: array containing the updated transfers
         - logratio: array containing the log-ratio of excess demand
    """
    # Calculate firms' demand and workers' supply
    nXpX = nX * probX(t_init) # Workers' supply to firms
    nYpY = nY * probY(t_init) # Firms' demand for workers

    # Calculate the log-ratio of firms' demand and workers' supply
    logratio = jnp.log(nYpY / nXpX)

    # Update transfer
    t_update = t_init + K * logratio
    return t_update, logratio

def EndogenousVariables(cX: jnp.ndarray, 
                        cY: jnp.ndarray, 
                        probX_vX: callable, 
                        probY_vY: callable, 
                        exog: Pytree,
                        accelerator: str = "None") -> Pytree:
    """Solves the matching model for a given specification of the choice
    probabilities (probX_vX) and (probY_vY) and exogenous variables (exog).

        Inputs:
          - cX: matrix describing the adjustment of the step lenght for workers
          - cY: matrix describing the adjustment of the step lenght for firms
          - probX_vX: workers' choice probabilities as function of payoffs, vX
          - probY_vY: firms' choice probabilities as function of payoffs, vY
          - exog: pytree containing the exogenous variables of the model

        Outputs:
          - endog: pytree containing the endogenous variables of the model
    """
    # Redefine choice probabilities as functions of transfer
    probX_T = lambda T: probX_vX((exog.utilityX + T) / exog.scaleX)
    probY_T = lambda T: probY_vY((exog.utilityY - T) / exog.scaleY)

    # Calculate the adjustment of the step length in the fixed point equation
    K = (cX * exog.scaleX * cY * exog.scaleY) / (cX * exog.scaleX + cY * exog.scaleY)

    # Set up system of fixed point equations
    fxp = lambda T: UpdateTransfer(T, K, exog.nX, exog.nY, probX_T, probY_T)

    # Initial guess for transfer
    transfer_init = jnp.zeros((exog.typesX, exog.typesY))

    @dataclass
    class Endog(Pytree, mutable=True):
        # Find the equilibrium transfer by fixed point iterations
        transfer = FixedPointRoot(fxp, transfer_init, acceleration=accelerator)[0]

        # Calculate the choice probabilities of the workers' (pX) and firms' (pY)
        probX = probX_T(transfer)
        probY = probY_T(transfer)

        # Calculate the choice probabilities for the outside options
        probX0 = 1 - jnp.sum(probX, axis=exog.axisX, keepdims=True)
        probY0 = 1 - jnp.sum(probY, axis=exog.axisY, keepdims=True)

        # Calculate the equilibrium distribution of the matches
        matched = exog.nX * probX

        # Calculate the unmatched workers and firms
        unmatchedX = exog.nX * probX0
        unmatchedY = exog.nY * probY0
    return Endog()
