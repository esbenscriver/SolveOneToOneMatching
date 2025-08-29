"""
test module/DiscreteChoiceModel.py
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from module.DiscreteChoiceModel import (
    LogitModel,
    NestedLogitModel,
    GeneralizedNestedLogitModel,
    ModelType,
)

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)


def assert_model_outside_option_True(model_X: ModelType, model_name: str) -> None:
    choice_probabilities_inside, choice_probabilities_outside = (
        model_X.ChoiceProbabilities(model_X.utility)
    )
    choice_probabilities_inside_sum = jnp.sum(
        choice_probabilities_inside, axis=model_X.axis, keepdims=True
    )
    choice_probabilities_sum = (
        choice_probabilities_inside_sum + choice_probabilities_outside
    )

    demand_outside = model_X.Demand_outside_option(model_X.utility)
    demand_inside = jnp.sum(
        model_X.Demand(model_X.utility), axis=model_X.axis, keepdims=True
    )
    check_demand_sum = demand_inside + demand_outside - model_X.n

    assert jnp.all(choice_probabilities_inside > 0.0), (
        f"{model_name}: Choice probabilities less than zero: {jnp.min(choice_probabilities_inside) = }"
    )
    assert jnp.all(choice_probabilities_inside < 1.0), (
        f"{model_name}: Choice probabilities bigger than one: {jnp.max(choice_probabilities_inside) = }"
    )

    assert jnp.all(choice_probabilities_outside > 0.0), (
        f"{model_name}: Choice probabilities (outside) less than zero: {jnp.min(choice_probabilities_outside) = }"
    )
    assert jnp.all(choice_probabilities_outside < 1.0), (
        f"{model_name}: Choice probabilities (outside) bigger than one: {jnp.max(choice_probabilities_outside) = }"
    )

    assert jnp.all(choice_probabilities_inside_sum < 1.0), (
        f"{model_name}: Sum of choice probabilities bigger than one: {jnp.max(choice_probabilities_inside_sum) = }"
    )
    assert jnp.allclose(choice_probabilities_sum, 1.0), (
        f"{model_name}: Sum of choice probabilities does not sum to one: {jnp.min(choice_probabilities_sum) = }, {jnp.max(choice_probabilities_sum) = }"
    )

    assert jnp.allclose(check_demand_sum, 0.0), (
        f"{model_name}: Sum of demand does not sum to n: {jnp.min(check_demand_sum) = }, {jnp.max(check_demand_sum) = }"
    )


def assert_model_outside_option_False(model_X: ModelType, model_name: str) -> None:
    choice_probabilities_inside, choice_probabilities_outside = (
        model_X.ChoiceProbabilities(model_X.utility)
    )
    choice_probabilities_inside_sum = jnp.sum(
        choice_probabilities_inside, axis=model_X.axis
    )

    demand_inside = jnp.sum(
        model_X.Demand(model_X.utility), axis=model_X.axis, keepdims=True
    )

    assert jnp.all(choice_probabilities_inside > 0.0), (
        f"{model_name}: Choice probabilities less than zero: {jnp.min(choice_probabilities_inside) = }"
    )
    assert jnp.all(choice_probabilities_inside < 1.0), (
        f"{model_name}: Choice probabilities bigger than one: {jnp.max(choice_probabilities_inside) = }"
    )

    assert jnp.allclose(choice_probabilities_outside, 0.0), (
        f"{model_name}: Sum of choice probabilities of outside option is not zero: {jnp.min(choice_probabilities_outside) = }, {jnp.max(choice_probabilities_outside) = }"
    )
    assert jnp.allclose(choice_probabilities_inside_sum, 1.0), (
        f"{model_name}: Sum of choice probabilities does not sum to one: {jnp.min(choice_probabilities_inside_sum) = }, {jnp.max(choice_probabilities_inside_sum) = }"
    )

    assert jnp.allclose(demand_inside, model_X.n), (
        f"{model_name}: Sum of demand does not sum to n: {jnp.min(demand_inside) = }, {jnp.max(demand_inside) = }"
    )


def test_LogitModel_outside_option_True() -> None:
    types_X, types_Y = 4, 6

    utility_X = random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
    scale_X = random.uniform(key=random.PRNGKey(112), shape=(types_X, 1))
    n_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1))

    model_X = LogitModel(utility=utility_X, scale=scale_X, n=n_X, outside_option=True)

    assert_model_outside_option_True(model_X, "Logit model")


def test_LogitModel_outside_option_False() -> None:
    types_X, types_Y = 4, 6

    utility_X = random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
    scale_X = random.uniform(key=random.PRNGKey(112), shape=(types_X, 1))
    n_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1))

    model_X = LogitModel(utility=utility_X, scale=scale_X, n=n_X, outside_option=False)

    assert_model_outside_option_False(model_X, "Logit model")


def test_NestedLogitModel_outside_option_True() -> None:
    types_X, types_Y = 4, 6

    utility_X = random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
    scale_X = random.uniform(key=random.PRNGKey(112), shape=(types_X, 1))
    n_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1))

    nests_Y = 3
    nest_parameter_X = random.uniform(key=random.PRNGKey(114), shape=(types_X, nests_Y))

    model_X = NestedLogitModel(
        utility=utility_X,
        scale=scale_X,
        nest_index=jnp.arange(types_Y) % nests_Y,
        nest_parameter=nest_parameter_X,
        n=n_X,
    )

    assert_model_outside_option_True(model_X, "Nested Logit model")


def test_GeneralizedNestedLogitModel_outside_option_True() -> None:
    types_X, types_Y = 4, 6

    utility_X = random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
    scale_X = random.uniform(key=random.PRNGKey(112), shape=(types_X, 1))
    n_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1))

    nests_Y = 3
    nest_share_X = random.dirichlet(
        key=random.PRNGKey(114), alpha=jnp.ones((nests_Y,)), shape=(types_Y,)
    )
    nest_parameter_X = random.uniform(key=random.PRNGKey(115), shape=(types_X, nests_Y))

    model_X = GeneralizedNestedLogitModel(
        utility=utility_X,
        scale=scale_X,
        nest_share=nest_share_X,
        nest_parameter=nest_parameter_X,
        n=n_X,
    )

    assert_model_outside_option_True(model_X, "Generalized nested logit model")
