"""
JAX implementation of fixed-point iteration algoritm to solve one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax.numpy as jnp

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass

# import fixed-point iterator
from fxp_jax import fxp_root

from SolveOneToOneMatching.DiscreteChoiceModel import ModelType


@dataclass
class Solution(Pytree, mutable=False):
    """Solution of matching model

    Attributes:
        transfer (jnp.ndarray): equilibrium transfer
        matches (jnp.ndarray): equilibrium number of matches
    """

    transfer: jnp.ndarray
    matches: jnp.ndarray


@dataclass
class MatchingModel(Pytree, mutable=False):
    """Matching model

    Attributes:
        model_X (ModelType): demand model for agents of type X
        model_Y (ModelType): demand model for agents of type Y
    """

    model_X: ModelType
    model_Y: ModelType

    @property
    def numberOfTypes_X(self) -> int:
        return self.model_X.n.size

    @property
    def numberOfTypes_Y(self) -> int:
        return self.model_Y.n.size

    @property
    def adjust_step(self) -> jnp.ndarray:
        scale_adjustment_X = self.model_X.scale * self.model_X.adjustment
        scale_adjustment_Y = self.model_Y.scale * self.model_Y.adjustment
        return (scale_adjustment_X * scale_adjustment_Y.T) / (
            scale_adjustment_X + scale_adjustment_Y.T
        )

    def Demand_X(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type X's demand for agents of type Y

        Args:
            transfer (jnp.ndarray): match-specific transfers

        Returns:
            demand (jnp.ndarray): match-specific demand

        """
        v_X = (self.model_X.utility + transfer) / self.model_X.scale
        return self.model_X.Demand(v_X)

    def Demand_Y(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type Y's demand for agents of type X

        Args:
            transfer (jnp.ndarray): match-specific transfers

        Returns:
            demand (jnp.ndarray): match-specific demand

        """
        v_Y = (self.model_Y.utility - transfer.T) / self.model_Y.scale
        return self.model_Y.Demand(v_Y).T

    def UpdateTransfers(
        self, t_initial: jnp.ndarray, adjust_step: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes excess demand and updates fixed point equation for transfers

        Args:
            t_initial (jnp.ndarray): initial transfers
            adjust_step (jnp.ndarray): adjustment terms for step lenght

        Returns:
        t_updated (jnp.ndarray):
            updated transfer.
        logratio (jnp.ndarray):
            log-ratio of excess demand.
        """
        # Calculate demand for both sides of the market
        demand_X = self.Demand_X(t_initial)  # type X's demand for type Y
        demand_Y = self.Demand_Y(t_initial)  # type Y's demand for type X

        # Calculate the log-ratio of excess demand for type X
        logratio = jnp.log(demand_Y / demand_X)

        # Update transfer
        t_updated = t_initial + adjust_step * logratio
        return t_updated, logratio

    def Solve(
        self,
        acceleration: str = "None",
        step_tol: float = 1e-10,
        root_tol: float = 1e-8,
        max_iter: int = 100_000,
    ) -> Solution:
        """Solve equilibrium transfers of matching model and store equilibrium outcomes

        Args:
            acceleration (str): set accelerator of fixed-point iterations ("None" or "SQUAREM)
            step_tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            root_tol (float): stopping tolerance for root size of fixed-point iterations, z_{i}
            max_iter (int): maximum number of iterations

        Returns:
            solution (Solution):
                solution of the model (transfer, matches)
        """

        # Initial guess for transfer
        transfer_init = jnp.zeros((self.numberOfTypes_X, self.numberOfTypes_Y))

        def fixed_point(t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return self.UpdateTransfers(t, self.adjust_step)

        # Find equilibrium transfer by fixed-point iterations
        result = fxp_root(
            fixed_point,
            step_tol=step_tol,
            root_tol=root_tol,
            max_iter=max_iter,
            accelerator=acceleration,
        ).solve(guess=transfer_init)
        return Solution(transfer=result.x, matches=self.Demand_X(result.x))
