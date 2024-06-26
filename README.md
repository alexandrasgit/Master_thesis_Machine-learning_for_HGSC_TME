# Uncovering Tumor Microenvironment features linked to clinico-molecular types of Ovarian cancer using Machine learning

### About the project
This repository is composed of the code for Random Forest pipelines and their results' analysis.

### Getting started

This repository includes two folders:

1. Pipelines
   * mol_profiles_bin_classification.py 
   * clinical_outcome_bin_classification.py 
  
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
   
   1. Functions' files
    * analysis_of_molprofiles_results.py 
    * analysis_of_OS_results.py
   
   2. Notebooks with created figures
    * Molecular_profiles_preds.ipynb
    * OS_preds.ipynb

***

### Contact
Aleksandra Shabanova aleksandra.shabanova@helsinki.fi

### Availability of pipeline 
The pipeline will be soon released for public use at farkkilab/FeatX (expected time is Summer 2024) and updated versions can be found there.
