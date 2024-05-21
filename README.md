# food-proportion-solver <img src="cake.png" alt="drawing" width="40" style="vertical-align:middle"/>

A Python tool to reverse engineer food quantities from an ingredient list and nutritional values.


**Example use case:** You really like this Lemon Cake from the grocery store and would like to make it at home.\
_Problem:_ You have the ingredients list but not their quantities. This tool attempts to recover the ingredients proportions by using the product nutritional values.

_This tool was made as a Human-LLM collaboration. The initial solution used Scipy (SLSQP) but had convergence issues, likely due to the problem being underconstrained (most of the times). The final solution uses CvxPy._


# Usage

## Getting started

We recommend creating a Python environment
```
python3 -m venv venv-food
source venv-food/bin/activate
```

Clone the repository
```
git clone https://github.com/paul-cayet/food-proportion-solver.git
```

Install the requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```



## Data set-up

### Food database

The food database contains nutritional values for the ingredients contained in the product. We have a simple example here, which gives the quantity in grams of each nutritional element for 100 grams (g) of ingredient. 

A database of ingredients nutritional values can be found from the [USDA website](https://fdc.nal.usda.gov/download-datasets.html).


### Target product information

⚠️ **Note: The tool recovers proportions for 100g of final product. Scale according to the desired quantity.**

To recover the ingredients proportions, we need the following information:

`final_product_nutritional` : The nutritional information for 100g of product.\
`ingredients` : The list of ingredients by decending proportion (which is how ingredients list are naturally ordered).\
`initial_guess` : (Optional) an initial guess for the food quantities (will be normalized to 100 g of product mass if not None).\
`known_quantities` : (Optional) to use if we know a proportion of one or several ingredients in the recipe (⚠️ the known quantities must be manually scaled to 100g of final product).


### Solving for the ingredient quantities

```
python3 solver.py -f food_database.csv -t target_food_ingredients.yaml
```

You can change the regularization parameters if you need to (in most cases the problem is underconstrained so regularization of the solution is helpful).