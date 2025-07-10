"""
JAX implementation of discrete choice models

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518v3)
"""
import jax.numpy as jnp

def Logit(v: jnp.ndarray, axis: int = 0, outside_option: bool = True) -> jnp.ndarray:
    """Calculates the logit choice probabilitie of the inside options

        Inputs:
         - v: choice-specific payoffs
         - axis: tells which dimensions to sum over

        Outputs:
         - choice probabilities of alternatives.
    """
    # exponentiated payoff of outside option
    expV0 = jnp.where(outside_option, 1.0, 0.0)

    # exponentiated payoffs of inside options (nominator)
    nominator = jnp.exp(v)

    # denominator of choice probabilities
    denominator = expV0 + jnp.sum(nominator, axis=axis, keepdims=True)
    return nominator / denominator

def GNLogit(
        v: jnp.ndarray, 
        degree: jnp.ndarray, 
        nesting: jnp.ndarray, 
        axis: int = 0, 
        outside_option: bool = True,
    ) -> jnp.ndarray:
    """Calculates the generalized nested logit choice probabilitie of the inside
    options

        Inputs:
         - v: choice-specific payoffs
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
    expV0 = jnp.where(outside_option, 1.0, 0.0)
    expV_nest = jnp.sum(expV, axis=axisA, keepdims=True)

    nominator = jnp.sum(expV * (expV_nest ** (nesting - 1.0)), axis=axisN)
    denominator = expV0 + jnp.sum(
        jnp.squeeze(expV_nest ** nesting),
        axis=axis,
        keepdims=True
    )
    return nominator / denominator

