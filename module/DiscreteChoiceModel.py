"""
JAX implementation of discrete choice models

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518v3)
"""
import jax.numpy as jnp
from simple_pytree import Pytree, dataclass

@dataclass
class LogitModel(Pytree, mutable=True):
    utility: jnp.ndarray
    scale: jnp.ndarray

    n: jnp.ndarray

    outside_option: bool = True
    axis: int = 1

    @property
    def adjustment(self) -> jnp.ndarray:
        # Set up adjustment factor for the logit model
        return jnp.ones_like(self.scale, dtype=float)

    def ChoiceProbabilities(self, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the logit choice probabilitie for inside and outside options."""
        v_max = jnp.max(v, axis=self.axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        expV_inside = jnp.exp(v - v_max)

        # if outside option exists exponentiate the centered payoff
        expV_outside = jnp.where(self.outside_option, jnp.exp(-v_max), jnp.zeros_like(v_max))

        # denominator of choice probabilities
        denominator = expV_outside + jnp.sum(expV_inside, axis=self.axis, keepdims=True)
        return expV_inside / denominator, expV_outside / denominator
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options."""
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option."""
        return self.n * self.ChoiceProbabilities(v)[1]

@dataclass
class NestedLogitModel(Pytree, mutable=True):
    utility: jnp.ndarray
    scale: jnp.ndarray

    nesting_index: jnp.ndarray
    nesting_parameter: jnp.ndarray

    n: jnp.ndarray

    outside_option: bool = True
    axis: int = 1

    @property
    def number_of_nests(self) -> int:
        return self.nesting_parameter.shape[1]

    @property
    def nesting_structure(self) -> jnp.ndarray:
        # Set up matrix that indicate which nest each alternative belongs to
        index_of_nests = jnp.arange(self.number_of_nests)
        nesting_structur = jnp.expand_dims(self.nesting_index, self.axis) == jnp.expand_dims(index_of_nests, 1 - self.axis)
        return nesting_structur.astype(float)
    
    @property
    def adjustment(self) -> jnp.ndarray:
        # Set up adjustment factor for the nested logit model
        return self.nesting_parameter @ self.nesting_structure.T

    def ChoiceProbabilities(self, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the nested logit choice probabilities for inside and outside options."""
        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nesting_parameter = jnp.einsum('nk, jk -> nj', self.nesting_parameter, self.nesting_structure)

        v_max = jnp.max(v, axis=self.axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        expV_inside = jnp.exp((v - v_max) / nesting_parameter)

        # if outside option exists exponentiate the centered payoff
        expV_outside = jnp.where(self.outside_option, jnp.exp(-v_max), 0.0)
        
        denominator_cond = jnp.einsum('nj, jk -> nk', expV_inside, self.nesting_structure)
        P_cond = expV_inside / jnp.einsum('nk, jk -> nj', denominator_cond, self.nesting_structure)

        nominator_nest = denominator_cond ** self.nesting_parameter
        denominator_nest = expV_outside + jnp.sum(nominator_nest, axis=self.axis, keepdims=True)
        P_nest = jnp.einsum('nk, jk -> nj', nominator_nest, self.nesting_structure) / denominator_nest

        P_outside = expV_outside / denominator_nest
        return P_cond * P_nest, P_outside
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options."""
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option."""
        return self.n * self.ChoiceProbabilities(v)[1]
@dataclass
class GeneralizedNestedLogitModel(Pytree, mutable=True):
    utility: jnp.ndarray
    scale: jnp.ndarray

    nesting_structure: jnp.ndarray
    nesting_parameter: jnp.ndarray

    n: jnp.ndarray

    outside_option: bool = True
    axis: int = 1
    
    @property
    def adjustment(self) -> jnp.ndarray:
        # Set up adjustment factor for the generalized nested logit model
        return jnp.min(self.nesting_parameter, axis=self.axis, keepdims=True)

    def ChoiceProbabilities(self, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the nested logit choice probabilities for inside and outside options."""

        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nesting_parameter = self.nesting_parameter[:,None,:]
        
        v_max = jnp.max(v, axis=self.axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        expV_inside = jnp.exp((v - v_max))

        # if outside option exists exponentiate the centered payoff
        expV_outside = jnp.where(self.outside_option, jnp.exp(-v_max), jnp.zeros_like(v_max))

        nominator_ni_k = jnp.einsum('nj, jk -> njk', expV_inside, self.nesting_structure) ** (1 / nesting_parameter)
        denominator_ni_k = jnp.einsum('njk -> nk', nominator_ni_k)
        P_ni_k = nominator_ni_k / denominator_ni_k[:,None,:]

        nominator_nk = denominator_ni_k ** self.nesting_parameter
        denominator_nk = expV_outside.squeeze() + jnp.einsum('nk -> n', nominator_nk)
        P_nk = nominator_nk / denominator_nk[:,None]

        P_inside = jnp.einsum('njk, nk -> nj', P_ni_k, P_nk)
        P_outside = expV_outside / denominator_nk[:,None]

        return P_inside, P_outside
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options."""
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option."""
        return self.n * self.ChoiceProbabilities(v)[1]

ModelType = LogitModel | NestedLogitModel | GeneralizedNestedLogitModel