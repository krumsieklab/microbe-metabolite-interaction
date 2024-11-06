import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    
# Implement the Meinshausen and Buhlmann method
# Does Lasso regression and returns the coefficients which are neighborhood
def mb_neighborhood_selection(data1, data2):
    "This step does a 2-way MB algorithm. Uses Lasso regression: L1 regularization to select features."
    # Taking the dimensions of the input dataset
    n1, m1 = data1.shape  # n observations, m variables
    n2, m2 = data2.shape  
    
    assert n1 == n2
    
    # Creating an empty matrix for results
    neighborhoods = pd.DataFrame(np.zeros((m1, m2)))
    
    # Renaming data2 as X (as predictors)
    X = data2
    
    # Looping over features (columns) in data1
    for i in range(m1):
        # Set current feature as response y
        y = data1.iloc[:, i]
        
        # Fit LASSO model
        # Random state 0 for reproducibility
        lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X, y)
        
        # Identify coefficients
        coef = lasso.coef_
        
        # Only keep non-zero coefficients
        nonzero_indices = np.where(coef != 0)[0]
        
        # Store coefficients (not just 1s)
        neighborhoods.iloc[i, nonzero_indices] = coef[nonzero_indices]

    return neighborhoods

# Does neighborhood selection 
# Logic: MIN/MAX 

def find_bipartite_neighborhood(neighborhood1, neighborhood2, value):
    """Return matrix using min/max logic.
     This step is doing feature selection by using neighborhood selection.
    """
    
    if value=="max":
        # Element-wise maximum between the absolute values of the microbiome and metabolite neighborhoods
        # Equivalent to OR logic
        interaction = np.maximum(np.abs(neighborhood1), np.abs(neighborhood2.T))
    elif value=="min":
        # Element-wise minimum between the microbiome and metabolite neighborhoods
        # Equivalent to AND logic
        interaction = np.minimum(np.abs(neighborhood1), np.abs(neighborhood2.T))
    else:
        raise ValueError("value must be either 'max' or 'min'")
    
    # Replace zeros with NaNs for easier interpretation (optional)
    interaction_matrix = pd.DataFrame(interaction, index=neighborhood1.index, columns=neighborhood1.columns)
    
    return interaction_matrix