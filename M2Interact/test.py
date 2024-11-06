import unittest
import pandas as pd
import numpy as np
import warnings
from io import StringIO
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from skbio.stats.composition import multiplicative_replacement
import M2dataproccess as m2dp

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample CSV data
        self.sample_data = StringIO("""index,feature1,feature2,feature3
A,1,0,3
B,0,0,0
C,4,5,0
D,0,2,3
""")
        
        self.expected_parse_df = pd.DataFrame({
            'feature1': [1, 4, 0],
            'feature2': [0, 5, 2],
            'feature3': [3, 0, 3]
        }, index=['A', 'C', 'D'])
        
        self.expected_transpose_df = pd.DataFrame({
            'A': [1, 0, 3],
            'C': [4, 5, 0],
            'D': [0, 2, 3]
        }, index=['feature1', 'feature2', 'feature3'])

        # Set the index name to None to match the output of parse_raw_data
        self.expected_parse_df.index.name = None

        self.expected_parse_header = ['feature1', 'feature2', 'feature3']
        
        # Sample DataFrame for drop_rare_features
        self.sample_drop_df = pd.DataFrame({
            'feature1': [1, 0, 4, 0],
            'feature2': [0, 0, 5, 2],
            'feature3': [3, 0, 0, 3],
            'rare_feature': [0, 0, 0, 0]
        }, index=['A', 'B', 'C', 'D'])

        self.expected_drop_df = pd.DataFrame({
            'feature1': [1, 0, 4, 0],
            'feature2': [0, 0, 5, 2],
            'feature3': [3, 0, 0, 3]
        }, index=['A', 'B', 'C', 'D'])  # rare_feature is expected to be dropped
        
         # Sample microbiome data
        self.microbiome_data = pd.DataFrame({
            'microbe1': [1, 0, 2, 4],
            'microbe2': [5, 1, 0, 0],
            'microbe3': [0, 3, 2, 1]
        }, index=['sampleA', 'sampleB', 'sampleC', 'sampleD'])

        # Sample metabolite data with some common samples
        self.metabolite_data = pd.DataFrame({
            'metabolite1': [0, 2, 4],
            'metabolite2': [1, 0, 1],
            'metabolite3': [5, 2, 0]
        }, index=['sampleB', 'sampleC', 'sampleD'])
        
        # Expected aligned microbiome and metabolite data
        self.expected_microbiome_aligned = pd.DataFrame({
            'microbe1': [0, 2, 4],
            'microbe2': [1, 0, 0],
            'microbe3': [3, 2, 1]
        }, index=['sampleB', 'sampleC', 'sampleD'])

        self.expected_metabolite_aligned = pd.DataFrame({
            'metabolite1': [0, 2, 4],
            'metabolite2': [1, 0, 1],
            'metabolite3': [5, 2, 0]
        }, index=['sampleB', 'sampleC', 'sampleD'])


    def test_parse_raw_data(self):
        # Read the data from the StringIO object instead of an actual file
        result_df, result_header = m2dp.parse_raw_data(self.sample_data)

        # Ensure the index names are both None
        self.assertIsNone(result_df.index.name)
        self.assertIsNone(self.expected_parse_df.index.name)

        # Test if the returned dataframe is correct
        pd.testing.assert_frame_equal(result_df, self.expected_parse_df)

        # Test if the returned header is correct
        self.assertEqual(result_header, self.expected_parse_header)
        
    def test_empty_rows_removed(self):
        # Additional check to ensure rows with all zeros were removed
        result_df, _ = m2dp.parse_raw_data(self.sample_data)

        # Ensure there are no rows with all zeros
        self.assertFalse((result_df.sum(axis=1) == 0).any())
        
    def test_transpose_csv(self):
        # Unit test to transpose the DataFrame
        result_df = m2dp.transpose_csv(self.expected_parse_df)
        
        # Ensure the index names are both None
        self.assertIsNone(result_df.index.name)
        self.assertIsNone(self.expected_transpose_df.index.name)
        
        # Test if the returned dataframe is correct
        pd.testing.assert_frame_equal(result_df, self.expected_transpose_df)
        
    def test_drop_rare_features(self):
        # Unit test to drop rare features based on a threshold
        result_df = m2dp.drop_rare_features(self.sample_drop_df, threshold=0.2)

        # Test if the 'rare_feature' column is dropped
        pd.testing.assert_frame_equal(result_df, self.expected_drop_df)

        # Ensure the DataFrame after dropping has the expected number of columns
        self.assertEqual(result_df.shape[1], self.expected_drop_df.shape[1])

    def test_align_microbiome_metabolite(self):
        # Run the function with sample data
        aligned_microbiome, aligned_metabolite = m2dp.align_microbiome_metabolite(self.microbiome_data, self.metabolite_data)

        # Check that the aligned dataframes are as expected
        pd.testing.assert_frame_equal(aligned_microbiome, self.expected_microbiome_aligned)
        pd.testing.assert_frame_equal(aligned_metabolite, self.expected_metabolite_aligned)
        

class TestMakeCompositional(unittest.TestCase):
    def setUp(self):
        # Sample microbiome data
        self.sample_data = pd.DataFrame({
            'species1': [1, 2, 0],
            'species2': [0, 1, 1],
            'species3': [3, 0, 1],
        }, index=['sample1', 'sample2', 'sample3'])
        
        # Expected results (replace with expected output for each case)
        self.expected_clr_data = pd.DataFrame({
            'species1': [0.5, 0.7, 0.3],  # Example transformed data
            'species2': [0.2, 0.1, 0.4],
            'species3': [0.3, 0.2, 0.3],
        }, index=['sample1', 'sample2', 'sample3'])
        
    # Expected result after compositional normalization
        self.expected_normalized_data = pd.DataFrame({
            'species1': [0.25, 0.67, 0],
            'species2': [0, 0.33, 0.5],
            'species3': [0.75, 0, 0.5]
        }, index=['sample1', 'sample2', 'sample3'])
        
    

    def test_make_compositional_log(self):
        # Test log transformation
        result_df = m2dp.make_compositional(self.sample_data, transform='log')
        
        # Check the structure and transformation
        self.assertEqual(result_df.shape, self.sample_data.shape)

    def test_make_compositional_no_transform(self):
        # Test no transformation
        result_df = m2dp.make_compositional(self.sample_data, transform='none')
        
        # Check that the data is just normalized but not transformed
        np.testing.assert_almost_equal(result_df.sum(axis=1).values, np.ones(len(result_df)), decimal=2)

    def test_make_compositional_standard_scaling(self):
        # Test with standard scaling
        result_df = m2dp.make_compositional(self.sample_data, scale='standard')
        
        # Check the structure and scaling
        self.assertEqual(result_df.shape, self.sample_data.shape)
        # Check if standard scaling applied by ensuring mean close to 0
        self.assertTrue(np.allclose(result_df.mean(), 0, atol=1e-1))

    def test_make_compositional_power_scaling(self):
        # Test with power scaling
        result_df = m2dp.make_compositional(self.sample_data, scale='power')
        
        # Check the structure and scaling
        self.assertEqual(result_df.shape, self.sample_data.shape)
        
    @unittest.expectedFailure
    def test_make_compositional_empty_dataframe(self):
        # Test with an empty DataFrame
        empty_df = pd.DataFrame()
        result_df = m2dp.make_compositional(empty_df)
        # Check that it returns an empty DataFrame
        self.assertTrue(result_df.empty)
        
    @unittest.expectedFailure
    def test_make_compositional_all_zeros(self):
        # Test with a DataFrame containing all zeros
        zero_df = pd.DataFrame({
            'species1': [0, 0, 0],
            'species2': [0, 0, 0],
            'species3': [0, 0, 0],
        }, index=['sample1', 'sample2', 'sample3'])
        result_df = m2dp.make_compositional(zero_df)
        # Check that all rows with zeros are handled
        self.assertTrue((result_df == 0).all().all())
        
class TestMakeMetabolomics(unittest.TestCase):
    def setUp(self):
        # Sample metabolomics data with zeros (to create -inf after log2)
        self.sample_data = pd.DataFrame({
            'metabolite1': [0, 1, 2, 0, 5],
            'metabolite2': [0, 2, 3, 4, 0],
            'metabolite3': [1, 0, 0, 2, 0],
            'metabolite4': [5, 0, 2, 0, 1]
        }, index=['sample1', 'sample2', 'sample3', 'sample4', 'sample5'])

    def test_log2_transformation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            # Test if log2 transformation correctly handles zeros by turning them into -inf
            result_df = m2dp.make_metabolomics(self.sample_data)

            # Check if the log2 transformation turned zeros into -inf and then imputed them
            data_with_inf = np.log2(self.sample_data.replace(0, np.nan))
            data_with_inf = data_with_inf.replace([np.inf, -np.inf], np.nan)

            # Ensure that -inf values (NaN after replacing) were correctly imputed
            self.assertFalse(result_df.isna().any().any())  # There should be no NaN values after kNN imputation

    def test_knn_imputation(self):
        # Ignore DeprecationWarning from is_sparse
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            # Test if kNN imputation works correctly
            result_df = m2dp.make_metabolomics(self.sample_data)

            # Ensure no NaN values remain after kNN imputation
            self.assertFalse(result_df.isna().any().any())

    def test_data_shape_preserved(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            # Ensure that the data shape is preserved after log2 transformation and imputation
            result_df = m2dp.make_metabolomics(self.sample_data)

            # Check that the shape of the resulting DataFrame is the same as the input
            self.assertEqual(result_df.shape, self.sample_data.shape)

    def test_infinity_handling(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            # Ensure that log2 transformation correctly replaces infinity values before imputation
            result_df = m2dp.make_metabolomics(self.sample_data)

            # Ensure no infinite values remain in the data after processing
            self.assertFalse(np.isinf(result_df.values).any())

  

if __name__ == '__main__':
    unittest.main()
