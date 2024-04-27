# Uncovering Tumor Microenvironment features linked to clinico-molecular types of Ovarian cancer using Machine learning

### About the project
This repository is composed of the code for Random Forest pipelines and their results' analysis.

### Prerequisites

To run the code:
1. Set up a conda environment with the help of nki_project.yml file

```
conda env create -f nki_project.yml
conda activate nki_env
```

### Getting started

This repository includes two folders:

1. Pipelines
   * mol_profiles_bin_classification.py 
   * clinical_outcome_bin_classification.py 
   * Conda environment is in nki_env.yml
  
    1. Prepare a CSV file with desired experiments

    | experiment  | channels | channels_to_transform | channels_to_outliers | channels_to_scale | types_of_cells | classes_column | classes_types | therapies | scaling_type | best_parameters | balanced_acc_train | balanced_acc_test | f1_train | f1_test | most_predictive_features | eliminated_features | permutation_scores | random_seed |
    | :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

    2. Don't forget to define the datasets that you will use: experiments CSV file and full cell dataset
    
    ```
    names = ['EXPERIMENTS_FILE_NAME.csv',...] 
    df = pd.read_csv("DATASET_FILENAME.csv")
    ```
    
    3. You are now ready to run the script

2. Analysis_of_results
    

***


### Contact
