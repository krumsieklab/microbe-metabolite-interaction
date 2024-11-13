import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from skbio.stats.composition import multiplicative_replacement, clr

    
def parse_raw_data(filename):
    """Return a pandas dataframe containing the raw data to classify.
    Data points should be in rows, features in columns. 
    """
    tbl = pd.read_csv(filename, index_col=0)
    header = list(tbl.columns.values)
    tbl = tbl.fillna(0)
    new_tbl = tbl[tbl.sum(axis=1) != 0]
    new_tbl.index.name = None
    return pd.DataFrame(new_tbl), header

# Transposing Data
def transpose_csv(filename):
    # Transpose the data 
    transposed_df = filename.transpose()
    return transposed_df
    


def drop_rare_features(df, threshold=0.2):
    """
    Drops features (microbial species/metabolites) based on a user defined threshold.
    Hint: If threshold is 0.2, anything with more than 80% zeros will be dropped.
    """
    # Convert the data to numeric, coercing errors to NaN, then fill NaN with zeros
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Calculate the number of samples (rows)
    num_samples = df.shape[0]
    
    # Calculate how many samples each microbe is present in (non-zero entries)
    present_in_samples = (df > 0).sum(axis=0)
    
    # Calculate the minimum number of samples required for a microbe to be retained
    min_required_samples = num_samples * threshold
    
    # Print the number of microbes before dropping
    print(f"Number of microbes before dropping: {df.shape[1]}")
    
    # Filter microbes (columns) that are present in at least `threshold` percent of samples
    filtered_df = df.loc[:, present_in_samples >= min_required_samples]
    
    # Print the number of microbes after dropping
    print(f"Number of microbes after dropping: {filtered_df.shape[1]}")
    
    return filtered_df

def drop_samples(filename):
    """Should we do this???
    Drop based on no of microbes/metabolites
    What if one sample have just 1 microbe/metabolite? Is it wrong? Biological or tech artifact?
    """
    return 0

def align_microbiome_metabolite(microbiome_data, metabolite_data):
    """
    Aligns microbiome and metabolite dataframes based on common sample names.
    """
    
    # Step 1: Find the common sample names between both dataframes
    common_samples = microbiome_data.index.intersection(metabolite_data.index)
    
    # Step 2: Reorder both dataframes based on the common sample names and drop missing samples
    microbiome_aligned = microbiome_data.loc[common_samples]
    metabolite_aligned = metabolite_data.loc[common_samples]
    
    return microbiome_aligned, metabolite_aligned

# Function for handling microbiome data and preprocessing  
def make_compositional(data, transform='clr', scale='none'):
    """
    Perform compositional normalization and transformation on microbiome data.
    Has multiple methods as option.
    Return a dataframe with transformed data.
    """   
 
    data_array = data.values

    # Step 1: Convert to relative abundances (Compositional normalization)
    row_sums = data_array.sum(axis=1, keepdims=True)
    data_rel = data_array / row_sums

    # Step 2: Perform multiplicative replacement to handle zero values in compositional data
    data_nonzero = multiplicative_replacement(data_rel)  

    # Step 3: Apply the selected transformation
    if transform == 'clr':
        data_trans = clr(data_nonzero)  # CLR transformation
        columns = data.columns  # Keep original columns

    elif transform == 'log':
        data_trans = np.log1p(data_nonzero)  # Log transformation (log(1 + x)) to handle zeros
        columns = data.columns

    else:
        data_trans = data_nonzero  # Default: no transformation
        columns = data.columns

    # Step 4: Apply scaling (if required)
    if scale == 'standard':
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_trans)
    elif scale == 'power':
        pt = PowerTransformer(method='yeo-johnson')
        normalized_data = pt.fit_transform(data_trans)  # Handles both positive and negative data
    else:
        normalized_data = data_trans  # No scaling
    
    # Step 5: Convert back to DataFrame with original indices and adjusted columns
    data_df = pd.DataFrame(normalized_data, index=data.index, columns=columns)

    return data_df

#For metabolomics dataset
#Convert data to log2 
def make_metabolomics(data):
    """
    This function processes a metabolomics dataset.
    It performs several steps including normalization, log transformation, and imputation.
    The method is similar to Maplet
    """
    
    # Step 1: Log2 transformation
    # Apply log2 transformation directly; zeros will turn into -inf
    data = np.log2(data)
    print("After log2 transformation:", data.shape)

    # Step 2: kNN Imputation (impute missing and infinity values)
    # Replace infinity values (-inf) resulting from log2(0) with NaN for imputation
    data = data.replace([np.inf, -np.inf], np.nan)
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data)
    print("After kNN imputation:", imputed_data.shape)
    
    # Convert the imputed data back to a DataFrame
    data_imputed = pd.DataFrame(imputed_data, index=data.index, columns=data.columns)
    print("After creating DataFrame from imputed data:", data_imputed.shape)
    
    return data_imputed


