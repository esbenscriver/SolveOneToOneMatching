import jax.numpy as jnp

# Import JAX code for logit and generalized nested logit choice proabilities
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