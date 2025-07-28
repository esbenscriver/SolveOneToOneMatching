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

from module.DiscreteChoiceModel import LogitModel, NestedLogitModel

import sys

@dataclass
class MatchingModel(Pytree, mutable=True):
    """ Matching model

        - Inputs:
            - model_X: demand model for agents of type X
            - model_Y: demand model for agents of type X
    """
    model_X: LogitModel|NestedLogitModel
    model_Y: LogitModel|NestedLogitModel

    transfer: jnp.ndarray|None = None
    matches: jnp.ndarray|None = None
    K: jnp.ndarray|None = None

    @property
    def numberOfTypes_X(self) -> int:
        return self.model_X.n.size
    
    @property
    def numberOfTypes_Y(self) -> int:
        return self.model_Y.n.size

    def _SetAdjustmentLength(self) -> None:
        """ Set the step length adjustment factor of the fixed-point iteration algorithm."""
        scale_adjustment_X = self.model_X.scale * self.model_X.adjustment
        scale_adjustment_Y = self.model_Y.scale * self.model_Y.adjustment

        self.K = (scale_adjustment_X * scale_adjustment_Y.T) / (scale_adjustment_X + scale_adjustment_Y.T)

    def _v_X(self, transfer: jnp.ndarray) -> jnp.ndarray:
        return (self.model_X.utility + transfer) / self.model_X.scale
    
    def _v_Y(self, transfer: jnp.ndarray) -> jnp.ndarray:
        return (self.model_Y.utility - transfer.T) / self.model_Y.scale

    def _Demand_X(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """ Computes choice probabilites of agents of type X."""
        return self.model_X.Demand(self._v_X(transfer))
        
    def _Demand_Y(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """ Computes choice probabilites of agents of type Y."""
        return self.model_Y.Demand(self._v_Y(transfer)).T

    def _UpdateTransfers(self, t_initial: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ Computes excess demand and updates fixed point equation for transfers

            Inputs:
            - t_initial: array containing initial transfers

            Outputs:
            - t_updated: array containing the updated transfers
            - logratio: array containing the log-ratio of excess demand
        """
        # Calculate demand for both sides of the market
        demand_X = self._Demand_X(t_initial) # type X's demand for type Y
        demand_Y = self._Demand_Y(t_initial) # type Y's demand for type X

        # Calculate the log-ratio of excess demand for type X
        logratio = jnp.log(demand_Y / demand_X)

        # Update transfer
        t_updated = t_initial + self.K * logratio
        return t_updated, logratio

    def Solve(self,
            acceleration: str = "None",
            step_tol: float = 1e-8,
            root_tol: float = 1e-6,
            max_iter: int = 100_000
        ) -> None:
        """ Solve equilibrium transfers of matching model and store equilibrium outcomes
        
            - Inputs:
                - acceleration (str): set accelerator of fixed-point iterations ("None" or "SQUAREM)
                - step_tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
                - root_tol (float): stopping tolerance for root size of fixed-point iterations, z_{i}
                - max_iter (int): maximum number of iterations
        """
        
        # Set adjustment length of fixed-point iterator
        self._SetAdjustmentLength()
        
        # Initial guess for transfer
        transfer_init = jnp.zeros((self.numberOfTypes_X, self.numberOfTypes_Y))

        # Find equilibrium transfer by fixed-point iterations
        self.transfer = FixedPointRoot(
            self._UpdateTransfers, 
            transfer_init, 
            acceleration=acceleration,
            step_tol=step_tol,
            root_tol=root_tol,
            max_iter=max_iter,
        )[0]

        self.matches = self._Demand_X(self.transfer)
