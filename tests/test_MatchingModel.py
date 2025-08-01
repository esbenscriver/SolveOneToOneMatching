"""
test module/MatchingModel.py
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

# import solver for one-to-one matching model
from module.MatchingModel import MatchingModel
from module.DiscreteChoiceModel import LogitModel, NestedLogitModel, GeneralizedNestedLogitModel

def assert_excess_demand(matching_model: MatchingModel, scenario: str) -> None:
    """ Assert excess demand. """
    if matching_model.transfer != None:
        demand_X = matching_model._Demand_X(matching_model.transfer)
        demand_Y = matching_model._Demand_Y(matching_model.transfer)
        assert jnp.allclose(demand_X, demand_Y), f"Error in scenario ({scenario}): {jnp.linalg.norm(demand_X - demand_Y) = }"

def test_solve() -> None:
    nests_X, nests_Y = 2, 3

    for types_X in [10, 100, 500]:
        for types_Y in [10, 100, 500]: 

            utility_X =-random.uniform(key=random.PRNGKey(111), shape=(types_X, types_Y))
            utility_Y = random.uniform(key=random.PRNGKey(211), shape=(types_Y, types_X))

            scale_X = random.uniform(key=random.PRNGKey(112), shape=(types_X, 1)) + 1.0
            scale_Y = random.uniform(key=random.PRNGKey(212), shape=(types_Y, 1)) + 1.0

            n_X = random.uniform(key=random.PRNGKey(113), shape=(types_X, 1))
            n_Y = random.uniform(key=random.PRNGKey(213), shape=(types_Y, 1)) + 1.0
            
            nesting_parameter_X = random.uniform(key=random.PRNGKey(114), shape=(types_X, nests_Y))
            nesting_structure_Y = random.dirichlet(key=random.PRNGKey(214), alpha=jnp.ones((nests_X,)), shape=(types_X,))

            nesting_parameter_Y = random.uniform(key=random.PRNGKey(115), shape=(types_Y, nests_X))
            nesting_structure_X = random.dirichlet(key=random.PRNGKey(215), alpha=jnp.ones((nests_Y,)), shape=(types_Y,))

            for acceleration in ['None','SQUAREM']:
                print('-----------------------------------------------------------------------')
                print(f"{types_X = }, {types_Y = }, {acceleration = }:")
                print('-----------------------------------------------------------------------')
                print('1. Solve a matching model with logit demand:')

                model_logit = MatchingModel(
                model_X = LogitModel(utility=utility_X, scale=scale_X, n=n_X),
                model_Y = LogitModel(utility=utility_Y, scale=scale_Y, n=n_Y),  
                )

                model_logit.Solve(acceleration=acceleration)
                scenario_logit = f"Logit model: {types_X = }, {types_Y = }, {acceleration = }"
                assert_excess_demand(model_logit, scenario_logit)

                print('-----------------------------------------------------------------------')
                print('2. Solve a matching model with nested logit demand:')

                model_nested_logit = MatchingModel(
                    model_X = NestedLogitModel(
                        utility=utility_X, 
                        scale=scale_X,

                        nesting_index=jnp.arange(types_Y) % nests_Y,
                        nesting_parameter=nesting_parameter_X,

                        n=n_X,
                    ),

                    model_Y = NestedLogitModel(
                        utility=utility_Y, 
                        scale=scale_Y,

                        nesting_index=jnp.arange(types_X) % nests_X,
                        nesting_parameter=nesting_parameter_Y,

                        n=n_Y,
                    ),
                )

                # model_nested_logit.Solve(acceleration=acceleration)
                # scenario_nested_logit = f"Nested logit model: {types_X = }, {types_Y = }, {acceleration = }"
                # assert_excess_demand(model_nested_logit, scenario_nested_logit)

                print('-----------------------------------------------------------------------')
                print('3. Solve a matching model with generalized nested logit demand:')

                model_GNL = MatchingModel(
                    model_X=GeneralizedNestedLogitModel(
                        utility=utility_X,
                        scale=scale_X,

                        nesting_structure=nesting_structure_X,
                        nesting_parameter=nesting_parameter_X,

                        n=n_X,
                    ),

                    model_Y = GeneralizedNestedLogitModel(
                        utility=utility_Y, 
                        scale=scale_Y,

                        nesting_structure=nesting_structure_Y,
                        nesting_parameter=nesting_parameter_Y,

                        n=n_Y,
                    ),
                )

                # model_GNL.Solve(acceleration=acceleration)
                # scenario_GNL = f"GNL model: {types_X = }, {types_Y = }, {acceleration = }"
                # assert_excess_demand(model_GNL, scenario_GNL)