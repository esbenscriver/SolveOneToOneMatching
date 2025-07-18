"""
JAX implementation of fixed-point iteration algoritm to solve one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518v3)
"""
import jax.numpy as jnp

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass

# import fixed-point iterator
from FixedPointJAX import FixedPointRoot

@dataclass
class ExogenousVariables(Pytree, mutable=False):
    # Set axis describing the alternatives for agents of type X and Y
    axisX: int
    axisY: int

    # Distribution of agents of type X and Y
    nX: jnp.ndarray
    nY: jnp.ndarray

    # Choice-specific utilities ofor agents of type X and Y
    utilityX: jnp.ndarray 
    utilityY: jnp.ndarray 

    # Scale parameters for agents of type X and Y
    scaleX: jnp.ndarray
    scaleY: jnp.ndarray

    @property
    def numberOfTypeX(self) -> int:
        return self.nX.size
    
    @property
    def numberOfTypeY(self) -> int:
        return self.nY.size

@dataclass
class EndogenousVariables(Pytree, mutable=True):
    # Find the equilibrium transfer by fixed point iterations
    transfers: jnp.ndarray

    # Choice probabilities for agents of type X and Y
    prob_matched_X: jnp.ndarray
    prob_matched_Y: jnp.ndarray

    # Choice probabilities for the outside options
    prob_unmatched_X: jnp.ndarray
    prob_unmatched_Y: jnp.ndarray

    # Equilibrium distribution of matched agents
    matched: jnp.ndarray

    # Equilibrium distribution of unmatched agents
    unmatched_X: jnp.ndarray
    unmatched_Y: jnp.ndarray

@dataclass
class MatchingModel(Pytree, mutable=True):
    exog: ExogenousVariables

    prob_X: callable
    prob_Y: callable

    cX: float = 1.0
    cY: float = 1.0

    endog: EndogenousVariables|None = None
    K: jnp.ndarray|None = None

    def _SetAdjustmentLength(self) -> None:
        """ Set the adjustment factor of the fixed-point iteration algorithm."""
        self.K = (self.cX * self.exog.scaleX * self.cY * self.exog.scaleY) / (self.cX * self.exog.scaleX + self.cY * self.exog.scaleY)

    def _prob_transfer_X(self, transfers: jnp.ndarray) -> jnp.ndarray:
        """ Calculates choice probabilites of agents of type X."""
        return self.prob_X((self.exog.utilityX + transfers) / self.exog.scaleX)
        
    def _prob_transfer_Y(self, transfers: jnp.ndarray) -> jnp.ndarray:
        """ Calculates choice probabilites of agents of type Y."""
        return self.prob_Y((self.exog.utilityY - transfers) / self.exog.scaleY)

    def _UpdateTransfers(self, t_initial: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates excess demand and updates fixed point equation for transfers

            Inputs:
            - t_initial: array containing initial transfers

            Outputs:
            - t_updated: array containing the updated transfers
            - logratio: array containing the log-ratio of excess demand
        """
        # Calculate demand for both sides of the market
        nXpX = self.exog.nX * self._prob_transfer_X(t_initial) # Demand of agents of type X
        nYpY = self.exog.nY * self._prob_transfer_Y(t_initial) # Demand of agents of type Y

        # Calculate the log-ratio of excess demand
        logratio = jnp.log(nYpY / nXpX)

        # Update transfer
        t_updated = t_initial + self.K * logratio
        return t_updated, logratio

    def Solve(self, acceleration: str = "None") -> None:
        """ Solve equilibrium transfers of matching model and store endogenous variables"""
        
        #Set adjustment length of fixed-point iterator
        self._SetAdjustmentLength()
        
        # Initial guess for transfer
        transfers_init = jnp.zeros((self.exog.numberOfTypeX, self.exog.numberOfTypeY))

        # Find equilibrium transfer by fixed-point iterations
        transfers = FixedPointRoot(self._UpdateTransfers, transfers_init, acceleration=acceleration)[0]

        # Calculate probabilities of agents being matched
        prob_matched_X = self._prob_transfer_X(transfers)
        prob_matched_Y = self._prob_transfer_Y(transfers)

        # Calculate probabilities of agents being unmatched
        prob_unmatched_X = 1 - jnp.sum(prob_matched_X, axis=self.exog.axisX, keepdims=True)
        prob_unmatched_Y = 1 - jnp.sum(prob_matched_Y, axis=self.exog.axisY, keepdims=True)

        # Calculate distribution of matched agents
        matched = self.exog.nX * prob_matched_X

        # Calculate distribution of unmatched agents
        unmatched_X = self.exog.nX * prob_unmatched_X
        unmatched_Y = self.exog.nY * prob_unmatched_Y

        # Store equilibrium outcomes in dataclass
        self.endog = EndogenousVariables(
            transfers=transfers,

            prob_matched_X=prob_matched_X,
            prob_matched_Y=prob_matched_Y,

            prob_unmatched_X=prob_unmatched_X,
            prob_unmatched_Y=prob_unmatched_Y,

            matched=matched,

            unmatched_X=unmatched_X,
            unmatched_Y=unmatched_Y
        )
