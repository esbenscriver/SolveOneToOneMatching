"""
JAX implementation of fixed-point iteration algoritm to solve one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax.numpy as jnp

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

from SolveOneToOneMatching.DiscreteChoiceModel import ModelType

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)


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
    def adjust_step(self) -> jnp.ndarray:
        scale_adjustment_X = self.model_X.scale * self.model_X.adjustment
        scale_adjustment_Y = self.model_Y.scale * self.model_Y.adjustment
        return (scale_adjustment_X * scale_adjustment_Y.T) / (
            scale_adjustment_X + scale_adjustment_Y.T
        )
    
    def Payoff_X(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type X's payoffs
        
        Args:
            transfer (jnp.ndarray): match-specific transfers
             
        returns:
            v_X (jnp.ndarray): match-specific payoffs 
        """
        return (self.model_X.utility + transfer) / self.model_X.scale
    
    def Payoff_Y(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type Y's payoffs
        
        Args:
            transfer (jnp.ndarray): match-specific transfers
             
        returns:
            v_Y (jnp.ndarray): match-specific payoffs 
        """
        return (self.model_Y.utility - transfer.T) / self.model_Y.scale

    def Demand_X(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type X's demand for agents of type Y

        Args:
            transfer (jnp.ndarray): match-specific transfers

        Returns:
            demand (jnp.ndarray): demand for inside options
        """
        v_X = self.Payoff_X(transfer)
        return self.model_X.n * self.model_X.ChoiceProbabilities(v_X)[0]
    
    def Demand_outside_X(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type X's demand for outside option

        Args:
            transfer (jnp.ndarray): match-specific transfers

        Returns:
            demand (jnp.ndarray): demand for outside option
        """
        v_X = self.Payoff_X(transfer)
        return self.model_X.n * self.model_X.ChoiceProbabilities(v_X)[1]

    def Demand_Y(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type Y's demand for agents of type X

        Args:
            transfer (jnp.ndarray): match-specific transfers

        Returns:
            demand (jnp.ndarray): demand for inside options
        """
        v_Y = self.Payoff_Y(transfer)
        return (self.model_Y.n * self.model_Y.ChoiceProbabilities(v_Y)[0]).T
    
    def Demand_outside_Y(self, transfer: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type Y's demand for outside option

        Args:
            transfer (jnp.ndarray): match-specific transfers

        Returns:
            demand (jnp.ndarray): demand for outside option
        """
        v_Y = (self.model_Y.utility - transfer.T) / self.model_Y.scale
        return (self.model_Y.n * self.model_Y.ChoiceProbabilities(v_Y)[1]).T

    def UpdateTransfers(
        self, t_initial: jnp.ndarray, adjust_step: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes excess demand and updates fixed point equation for transfers

        Args:
            t_initial (jnp.ndarray): initial transfers
            adjust_step (jnp.ndarray): adjustment terms for step lenght

        Returns:
            t_updated (jnp.ndarray): updated transfers
        """
        # Calculate demand for both sides of the market
        demand_X = self.Demand_X(t_initial)  # type X's demand for type Y
        demand_Y = self.Demand_Y(t_initial)  # type Y's demand for type X

        # Update transfer
        t_updated = t_initial + adjust_step * jnp.log(demand_Y / demand_X)
        return t_updated

    def Solve(
        self,
        fixed_point_solver: SolverTypes = FixedPointIteration,
        tol: float = 1e-10,
        maxiter: int = 1_000,
        verbose: bool = True,
    ) -> Solution:
        """Solve equilibrium transfers of matching model and store equilibrium outcomes

        Args:
            fixed_point_solver (SolverTypes): solver used for solving fixed point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
            step_tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            max_iter (int): maximum number of iterations

        Returns:
            solution (Solution): solution of the model (transfer, matches)
        """
        # Initial guess for equilibrium transfers
        transfer_init = jnp.zeros((self.model_X.n.size, self.model_Y.n.size))

        # Find equilibrium transfers
        result = fixed_point_solver(
            self.UpdateTransfers,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        ).run(transfer_init, self.adjust_step)
        return Solution(transfer=result.params, matches=self.Demand_X(result.params))
