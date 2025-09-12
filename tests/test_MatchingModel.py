"""
test SolveOneToOneMatching/MatchingModel.py
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

# import solver for one-to-one matching model
from SolveOneToOneMatching.MatchingModel import MatchingModel, SolverTypes
from SolveOneToOneMatching.DiscreteChoiceModel import (
    LogitModel,
    NestedLogitModel,
    GeneralizedNestedLogitModel,
)

from typing import Literal
import pytest

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "types_X, types_Y, model_name_X, model_name_Y, fixed_point_solver",
    [
        (100, 200, "Logit", "Logit", FixedPointIteration),
        (100, 200, "Logit", "Logit", AndersonAcceleration),
        (100, 200, "Logit", "Logit", SquaremAcceleration),
        (300, 200, "Logit", "Logit", SquaremAcceleration),
        (500, 700, "Logit", "Logit", SquaremAcceleration),
        (100, 200, "Logit", "NestedLogit", SquaremAcceleration),
        (100, 200, "Logit", "GeneralizedNestedLogit", SquaremAcceleration),
        (100, 200, "NestedLogit", "Logit", SquaremAcceleration),
        (100, 200, "NestedLogit", "NestedLogit", SquaremAcceleration),
        (100, 200, "NestedLogit", "GeneralizedNestedLogit", SquaremAcceleration),
        (100, 200, "GeneralizedNestedLogit", "Logit", SquaremAcceleration),
        (100, 200, "GeneralizedNestedLogit", "NestedLogit", SquaremAcceleration),
        (
            100,
            200,
            "GeneralizedNestedLogit",
            "GeneralizedNestedLogit",
            SquaremAcceleration,
        ),
    ],
)
def test_solve(
    types_X: int,
    types_Y: int,
    model_name_X: Literal["Logit", "NestedLogit", "GeneralizedNestedLogit"],
    model_name_Y: Literal["Logit", "NestedLogit", "GeneralizedNestedLogit"],
    fixed_point_solver: SolverTypes,
) -> None:
    nests_X, nests_Y = 2, 3

    utility_X = -random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
    utility_Y = random.uniform(key=random.PRNGKey(211), shape=(types_Y, types_X))

    scale_X = random.uniform(key=random.PRNGKey(112), shape=(types_X, 1)) + 1.0
    scale_Y = random.uniform(key=random.PRNGKey(212), shape=(types_Y, 1)) + 1.0

    n_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1))
    n_Y = random.uniform(key=random.PRNGKey(213), shape=(types_Y, 1)) + 1.0

    nest_parameter_X = random.uniform(
        key=random.PRNGKey(114),
        shape=(types_X, nests_Y),
        minval=0.1,
        maxval=1.0,
    )
    nest_share_Y = random.dirichlet(
        key=random.PRNGKey(214), alpha=jnp.ones((nests_X,)), shape=(types_X,)
    )

    nest_parameter_Y = random.uniform(
        key=random.PRNGKey(115),
        shape=(types_Y, nests_X),
        minval=0.1,
        maxval=1.0,
    )
    nest_share_X = random.dirichlet(
        key=random.PRNGKey(215), alpha=jnp.ones((nests_Y,)), shape=(types_Y,)
    )

    if model_name_X == "Logit":
        model_X = LogitModel(utility=utility_X, scale=scale_X, n=n_X)
    elif model_name_X == "NestedLogit":
        model_X = NestedLogitModel(
            utility=utility_X,
            scale=scale_X,
            nest_index=jnp.arange(types_Y) % nests_Y,
            nest_parameter=nest_parameter_X,
            n=n_X,
        )
    elif model_name_X == "GeneralizedNestedLogit":
        model_X = GeneralizedNestedLogitModel(
            utility=utility_X,
            scale=scale_X,
            nest_share=nest_share_X,
            nest_parameter=nest_parameter_X,
            n=n_X,
        )

    if model_name_Y == "Logit":
        model_Y = LogitModel(utility=utility_Y, scale=scale_Y, n=n_Y)
    elif model_name_Y == "NestedLogit":
        model_Y = NestedLogitModel(
            utility=utility_Y,
            scale=scale_Y,
            nest_index=jnp.arange(types_X) % nests_X,
            nest_parameter=nest_parameter_Y,
            n=n_Y,
        )
    elif model_name_Y == "GeneralizedNestedLogit":
        model_Y = GeneralizedNestedLogitModel(
            utility=utility_Y,
            scale=scale_Y,
            nest_share=nest_share_Y,
            nest_parameter=nest_parameter_Y,
            n=n_Y,
        )

    matching_model = MatchingModel(model_X=model_X, model_Y=model_Y)
    solution = matching_model.Solve(fixed_point_solver=fixed_point_solver)

    demand_X = matching_model.Demand_X(solution.transfer)
    demand_Y = matching_model.Demand_Y(solution.transfer)
    assert jnp.allclose(demand_X, demand_Y), (
        f"{jnp.linalg.norm(demand_X - demand_Y) = }"
    )
