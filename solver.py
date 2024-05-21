import numpy as np
import cvxpy as cp
import pandas as pd
from argparse import ArgumentParser
import yaml


def print_nutritional_errors(nutritional_db, product_nutritional_values, estimated_quantities):
    print(f'-----------------------\nIngredients\n-----------------------')
    print({k: round(v, 2) for k, v in estimated_quantities.items()})
    print(f'-----------------------\nNutritional values - estimated vs real\n-----------------------')
    for nut_type, nut_val in product_nutritional_values.items():
        nut_computed = np.sum([nutritional_db[ing_name][nut_type]/100 * ing_quantity for ing_name, ing_quantity in estimated_quantities.items()])
        print(f"{nut_type}: {nut_val:.2f} (estimated {nut_computed:.2f})")


def solve_quantities_cvxpy(ingredients, nutritional_db, nutrition_keys, 
                           final_product_nutritional, known_quantities, 
                           lambda_reg=0.1, lambda_relax=1.0, lambda_weight=1.0, initial_guess=None):
    n = len(ingredients)
    m = len(nutrition_keys)

    # Variables for quantities of each ingredient
    quantities = cp.Variable(n)

    if initial_guess is not None:
        quantities.value = initial_guess

    # Nutritional constraints matrix
    A_eq = np.zeros((m, n))
    b_eq = np.zeros(m)

    for i, key in enumerate(nutrition_keys):
        for j, ingredient in enumerate(ingredients):
            A_eq[i, j] = nutritional_db[ingredient][key] / 100
        b_eq[i] = final_product_nutritional[key]

    # Known weight constraints matrix
    A_known = np.zeros((len(known_quantities), n))
    b_known = np.zeros(len(known_quantities))
    
    for i, (ingredient, quantity) in enumerate(known_quantities.items()):
        j = ingredients.index(ingredient)
        A_known[i, j] = 1
        b_known[i] = quantity

    # Inequality constraints for decreasing quantities
    A_ub = np.zeros((n - 1, n))
    b_ub = np.zeros(n - 1)
    for i in range(n - 1):
        A_ub[i, i] = 1
        A_ub[i, i + 1] = -1

    # Constraints
    constraints = [
        A_ub @ quantities >= b_ub,  # Decreasing quantities
        quantities >= 0             # Non-negativity
    ]

    # Objective function: Minimize discrepancies and regularization
    nutrition_diff = sum(
        (A_eq[i, :] @ quantities - b_eq[i]) ** 2
        for i in range(m)
    )
    known_quantities_diff = sum(
        (A_known[i, :] @ quantities - b_known[i]) ** 2
        for i in range(len(known_quantities))
    )

    # Relaxed constraint for total weight equal to 100 grams (per 100g portion)
    total_weight_diff = (cp.sum(quantities) - 100) ** 2

    regularization = lambda_reg * cp.sum_squares(quantities)
    objective = cp.Minimize(lambda_relax * nutrition_diff + lambda_weight * (known_quantities_diff + total_weight_diff) + regularization)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start=True)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        estimated_quantities = dict(zip(ingredients, quantities.value))
        print_nutritional_errors(nutritional_db, final_product_nutritional, estimated_quantities)
        return estimated_quantities
    else:
        raise ValueError(f"Optimization problem failed: {problem.status}")


if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument("-f", "--food_database_path")
    parser.add_argument("-t", "--target_food_information")
    parser.add_argument("--lambda_reg", default=0.01, help="L2 regularization")
    parser.add_argument("--lambda_relax", default=1.0, help="nutrition regularization param")
    parser.add_argument("--lambda_weight", default=1.0, help="weight regularization param")
    args = parser.parse_args()

    # loading food data
    food_database = pd.read_csv(args.food_database_path,index_col=0)
    nutritional_db = food_database.T.to_dict()
    nutrition_keys = list(food_database.columns)

    # loading target product information
    target_food_info = yaml.safe_load(open(args.target_food_information,'r'))

    final_product_nutritional = target_food_info['final_product_nutritional']
    ingredients = target_food_info['ingredients']
    known_quantities = target_food_info['known_quantities']

    initial_guess = None
    if target_food_info['initial_guess'] is not None:
        initial_guess = np.array(target_food_info['initial_guess'])
        assert len(initial_guess) == len(ingredients), f"initial guess length ({len(initial_guess)=}) must be equal to ingredients length {len(ingredients)=}"
        initial_guess = initial_guess * (100 / initial_guess.sum()) # Scale initial guess to 100 grams

    estimated_quantities = solve_quantities_cvxpy(
        ingredients,
        nutritional_db,
        nutrition_keys,
        final_product_nutritional,
        known_quantities,
        lambda_reg=args.lambda_reg,
        lambda_relax=args.lambda_relax,
        lambda_weight=args.lambda_weight,
        initial_guess=initial_guess
    )

    print("\nEstimated Ingredient Proportions (grams per 100 grams of product):")
    for ingredient, quantity in estimated_quantities.items():
        print(f"{ingredient}: {quantity:.2f} grams")