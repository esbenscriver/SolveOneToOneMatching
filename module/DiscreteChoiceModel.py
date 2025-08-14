"""
JAX implementation of discrete choice models

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""
import jax.numpy as jnp
from simple_pytree import Pytree, dataclass

@dataclass
class LogitModel(Pytree, mutable=False):
    """ Logit discrete choice model

        Attributes:
            utility (jnp.ndarray): choice-specific utilities
            scale (jnp.ndarray): scale parameter
            n (jnp.ndarray): distribution of agents
            outside_option (bool): indicator for whether outside option is included
            axis (int): axis that defines choices

    """
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
        """ Compute the logit choice probabilities for inside and outside options
        
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
            P_inside (jnp.ndarray): 
                choice probabilities of inside options.
            P_outside (jnp.ndarray): 
                choice probabilities of outside option.
        """
        v_max = jnp.max(v, axis=self.axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        expV_inside = jnp.exp(v - v_max)

        # if outside option exists exponentiate the centered payoff
        expV_outside = jnp.where(self.outside_option, jnp.exp(-v_max), jnp.zeros_like(v_max))

        # denominator of choice probabilities
        denominator = expV_outside + jnp.sum(expV_inside, axis=self.axis, keepdims=True)
        return expV_inside / denominator, expV_outside / denominator
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options
            
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
                demand (jnp.ndarray): choice-specific demand

        """
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option
            
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
                demand (jnp.ndarray): demand for outside option

        """
        return self.n * self.ChoiceProbabilities(v)[1]

@dataclass
class NestedLogitModel(Pytree, mutable=False):
    """ Nested logit discrete choice model

        Attributes:
            utility (jnp.ndarray): choice-specific utilities
            scale (jnp.ndarray): scale parameter
            nest_index (jnp.ndarray): index of nest that the alternatives belong to
            nest_parameter (jnp.ndarray): nesting parameter
            n (jnp.ndarray): distribution of agents
            outside_option (bool): indicator for whether outside option is included
            axis (int): axis that defines choices

    """
    utility: jnp.ndarray
    scale: jnp.ndarray

    nest_index: jnp.ndarray
    nest_parameter: jnp.ndarray

    n: jnp.ndarray

    outside_option: bool = True
    axis: int = 1

    @property
    def number_of_nests(self) -> int:
        return self.nest_parameter.shape[1]

    @property
    def nest_structure(self) -> jnp.ndarray:
        # Set up matrix that indicate which nest each alternative belongs to
        index_of_nests = jnp.arange(self.number_of_nests)
        nesting_structur = jnp.expand_dims(self.nest_index, self.axis) == jnp.expand_dims(index_of_nests, 1 - self.axis)
        return nesting_structur.astype(float)
    
    @property
    def adjustment(self) -> jnp.ndarray:
        # Set up adjustment factor for the nested logit model
        return self.nest_parameter @ self.nest_structure.T

    def ChoiceProbabilities(self, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ Compute the nested logit choice probabilities for inside and outside options
        
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
            P_inside (jnp.ndarray): 
                choice probabilities of inside options.
            P_outside (jnp.ndarray): 
                choice probabilities of outside option.
        """
        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nest_parameter = jnp.einsum('nk, jk -> nj', self.nest_parameter, self.nest_structure)

        v_max = jnp.max(v, axis=self.axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        expV_inside = jnp.exp((v - v_max) / nest_parameter)

        # if outside option exists exponentiate the centered payoff
        expV_outside = jnp.where(self.outside_option, jnp.exp(-v_max), 0.0)
        
        denominator_cond = jnp.einsum('nj, jk -> nk', expV_inside, self.nest_structure)
        P_cond = expV_inside / jnp.einsum('nk, jk -> nj', denominator_cond, self.nest_structure)

        nominator_nest = denominator_cond ** self.nest_parameter
        denominator_nest = expV_outside + jnp.sum(nominator_nest, axis=self.axis, keepdims=True)
        P_nest = jnp.einsum('nk, jk -> nj', nominator_nest, self.nest_structure) / denominator_nest

        P_outside = expV_outside / denominator_nest
        return P_cond * P_nest, P_outside
    
    def ChoiceProbabilities_new(self, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ Compute the nested logit choice probabilities for inside and outside options
        
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
            P_inside (jnp.ndarray): 
                choice probabilities of inside options.
            P_outside (jnp.ndarray): 
                choice probabilities of outside option.
        """
        from jax.ops import segment_max

        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nest_parameter = jnp.einsum('nk, jk -> nj', self.nest_parameter, self.nest_structure)

        v_nest_parameter = v / nest_parameter

        # Step 4: subtract max per nest from each row
        centering_cond = segment_max(
            v_nest_parameter.T, 
            self.nest_index, 
            num_segments=self.number_of_nests
        )[self.nest_index,:].T

        # exponentiated centered payoffs of inside options
        expV_cond = jnp.exp(v_nest_parameter - centering_cond)
        
        denominator_cond = jnp.einsum('nj, jk -> nk', expV_cond, self.nest_structure)
        P_cond = expV_cond / jnp.einsum('nk, jk -> nj', denominator_cond, self.nest_structure)

        centering_nest = jnp.max(v_nest_parameter, axis=self.axis, keepdims=True)
        expV_inside = jnp.exp(v_nest_parameter - centering_nest)
        expV_outside = jnp.where(self.outside_option, jnp.exp(-centering_nest), 0.0)

        nominator_nest = jnp.einsum('nj, jk -> nk', expV_inside, self.nest_structure) ** self.nest_parameter
        denominator_nest = expV_outside + jnp.sum(nominator_nest, axis=self.axis, keepdims=True)
        P_nest = jnp.einsum('nk, jk -> nj', nominator_nest, self.nest_structure) / denominator_nest

        P_outside = expV_outside / denominator_nest
        return P_cond * P_nest, P_outside
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options
            
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
                demand (jnp.ndarray): choice-specific demand

        """
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option
            
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
                demand (jnp.ndarray): demand for outside option

        """
        return self.n * self.ChoiceProbabilities(v)[1]
@dataclass
class GeneralizedNestedLogitModel(Pytree, mutable=False):
    """ Generalized nested logit discrete choice model

        Attributes:
            utility (jnp.ndarray): choice-specific utilities
            scale (jnp.ndarray): scale parameter
            nest_share (jnp.ndarray): share that the alternatives belong to each nest
            nest_parameter (jnp.ndarray): nesting parameter
            n (jnp.ndarray): distribution of agents
            outside_option (bool): indicator for whether outside option is included
            axis (int): axis that defines choices

    """
    utility: jnp.ndarray
    scale: jnp.ndarray

    nest_share: jnp.ndarray
    nest_parameter: jnp.ndarray

    n: jnp.ndarray

    outside_option: bool = True
    axis: int = 1
    
    @property
    def adjustment(self) -> jnp.ndarray:
        # Set up adjustment factor for the generalized nested logit model
        return jnp.min(self.nest_parameter, axis=self.axis, keepdims=True)

    def ChoiceProbabilities(self, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ Compute the generalized nested logit choice probabilities for inside and outside options
        
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
            P_inside (jnp.ndarray): 
                choice probabilities of inside options.
            P_outside (jnp.ndarray): 
                choice probabilities of outside option.
        """

        # Explanation einsum indexes:
        # - n: index for agents' types
        # - j: index for alternatives (inside options)
        # - k: index for nests of alternatives (outside option is assumed to belong to its own nest)

        nest_parameter = self.nest_parameter[:,None,:]
        
        v_max = jnp.max(v, axis=self.axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        expV_inside = jnp.exp((v - v_max))

        # if outside option exists exponentiate the centered payoff
        expV_outside = jnp.where(self.outside_option, jnp.exp(-v_max), jnp.zeros_like(v_max))

        nominator_ni_k = jnp.einsum('nj, jk -> njk', expV_inside, self.nest_share) ** (1 / nest_parameter)
        denominator_ni_k = jnp.einsum('njk -> nk', nominator_ni_k)
        P_ni_k = nominator_ni_k / denominator_ni_k[:,None,:]

        nominator_nk = denominator_ni_k ** self.nest_parameter
        denominator_nk = expV_outside.squeeze() + jnp.einsum('nk -> n', nominator_nk)
        P_nk = nominator_nk / denominator_nk[:,None]

        P_inside = jnp.einsum('njk, nk -> nj', P_ni_k, P_nk)
        P_outside = expV_outside / denominator_nk[:,None]

        return P_inside, P_outside
    
    def Demand(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for inside options
            
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
                demand (jnp.ndarray): choice-specific demand

        """
        return self.n * self.ChoiceProbabilities(v)[0]

    def Demand_outside_option(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute demand for outside option
            
            Args:
                v (jnp.ndarray): choice-specific payoffs

            Returns:
                demand (jnp.ndarray): demand for outside option

        """
        return self.n * self.ChoiceProbabilities(v)[1]

ModelType = LogitModel | NestedLogitModel | GeneralizedNestedLogitModel