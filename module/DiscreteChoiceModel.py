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
        """Compute the logit choice probabilitie for the inside options."""
        expV0 = jnp.where(self.outside_option, 1.0, 0.0)

        # exponentiated payoffs of inside options (nominator)
        nominator = jnp.exp(v)

        # denominator of choice probabilities
        denominator = expV0 + jnp.sum(nominator, axis=self.axis, keepdims=True)
        return nominator / denominator, expV0 / denominator
    
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
        """Compute the nested logit choice probabilities for inside options."""
        expV0 = jnp.where(self.outside_option, 1.0, 0.0)

        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nesting_parameter = jnp.einsum('nk, jk -> nj', self.nesting_parameter, self.nesting_structure)
        
        nominator_cond = jnp.exp(v / nesting_parameter) 
        denominator_cond = jnp.einsum('nj, jk -> nk', nominator_cond, self.nesting_structure)
        P_cond = nominator_cond / jnp.einsum('nk, jk -> nj', denominator_cond, self.nesting_structure)

        nominator_nest = denominator_cond ** self.nesting_parameter
        denominator_nest = expV0 + jnp.sum(nominator_nest, axis=self.axis, keepdims=True)
        P_nest = jnp.einsum('nk, jk -> nj', nominator_nest, self.nesting_structure) / denominator_nest

        P_outside = expV0 / denominator_nest
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
        """Compute the nested logit choice probabilities for inside options."""
        expV0 = jnp.where(self.outside_option, 1.0, 0.0)

        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nesting_parameter = jnp.einsum('nk, jk -> nj', self.nesting_parameter, self.nesting_structure)
        
        nominator_cond = jnp.exp(v / nesting_parameter) 
        denominator_cond = jnp.einsum('nj, jk -> nk', nominator_cond, self.nesting_structure)
        P_cond = nominator_cond / jnp.einsum('nk, jk -> nj', denominator_cond, self.nesting_structure)

        nominator_nest = denominator_cond ** self.nesting_parameter
        denominator_nest = expV0 + jnp.sum(nominator_nest, axis=self.axis, keepdims=True)
        P_nest = jnp.einsum('nk, jk -> nj', nominator_nest, self.nesting_structure) / denominator_nest

        P_outside = expV0 / denominator_nest
        return P_cond * P_nest, P_outside
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options."""
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option."""
        return self.n * self.ChoiceProbabilities(v)[1]

ModelType = LogitModel | NestedLogitModel | GeneralizedNestedLogitModel