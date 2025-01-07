# microbe-metabolite-interaction


Installing:

```
cd microbe-metabolite-interaction/M2Interact
conda env create -f environment.yaml
```

OR

```
pip install -r requirement.txt
```

Running test
```
python test.py
```

Expected test output:
```
Ran 15 tests in 0.173s

OK (expected failures=2)
```

**Steps for running TwinsUK data:**

Microbiome data (CSV file with m1*n values):

1. Data should be either relative abundance or absolute value with no NAN/infinity values
2. Missing values should be recorded as 0
3. Data should be a csv file with microbes as row and samples as column (Eg: https://github.com/krumsieklab/microbe-metabolite-interaction/blob/main/M2Interact/biocrust/data/microbiome.csv)

Metabolite data (CSV file with m2*n values):

1. Data should be processed (cleaned, log-transformed) with no NAN/infinity values 
2. Rows should be metabolite and columns should be samples (Eg: https://github.com/krumsieklab/microbe-metabolite-interaction/blob/main/M2Interact/biocrust/data/metabolome.csv)

Code:
1. Please run the code present in https://github.com/krumsieklab/microbe-metabolite-interaction/blob/main/M2Interact/twinsuk.ipynb
2. The data location need to be updated in the first box
3. You should not change/update any other code blocks or files
4. The code is in form of a python notebook, extension for running it has to be installed (Eg: Jupyter Notebook, IPY kernel in VS Code)
5. The code also requires to install packages either which are present in requirement.txt/environment.yaml

Expected Output (On running succesfully this will be generated):
1. Refer to the folder https://github.com/krumsieklab/microbe-metabolite-interaction/tree/main/M2Interact/twinsuk
2. We will have distribution graphs of metabolite and 4 preprocessing on microbiome
3. The result folder will have 4 threshold and each will have 8 separate csv files
