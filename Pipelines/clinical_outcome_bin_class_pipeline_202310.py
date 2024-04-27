# Logs
import logging
# Configure logging
logging.basicConfig(filename='exp.out', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Handling dataframe
import numpy as np
import pandas as pd

# Pre-process data
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Preprocess features
from sklearn.preprocessing import LabelEncoder

# Load models
from sklearn.ensemble import RandomForestClassifier

# Model tuning 
from skopt.space import Integer
from skopt.utils import use_named_args
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from skopt import gp_minimize
from time import time

# Model testing
from sklearn.metrics import f1_score, balanced_accuracy_score, auc

# Feature selection 
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

import os

# multiprocessing
import multiprocessing as mp
from loky import get_reusable_executor

class Model:
    def __init__(self, exp_name = None, channels = None, channels_to_transform = None, channels_to_outliers = None, channels_to_scale = None, cells = None, target = None, therapies = None, mol_profiles = None, scaler = None, df = None, random_seed = None):

        self.rng = random_seed

        # Variables for data pre-processing
        self.df = df

        # from experiments.csv:
        self.exp_name = exp_name # Experiment column
        
        # Set the output directory and filename for the figures
        self.output_dir = os.path.dirname(os.path.abspath(__file__))

        # Collect required variables from csv file
        self.features = channels.split(",") # channels column
        self.types_of_cells = cells # types_of_cells column
        self.target = target # classes_types column
        self.mol_profiles = mol_profiles.split(",") # mol_profiles column
        self.therapies = therapies # therapies column 

        self.features_to_transform = channels_to_transform.split(",")
        self.features_for_outliers = channels_to_outliers.split(",")
        self.features_to_scale = channels_to_scale.split(",")

        # Define transformation type. Available types: LOG, LOG2, BOXCOX
        self.transformation = 'LOG2' 
        
        # Define how to remove outliers. Available options: trim_by_slide & remove
        self.operation = 'remove'
        
        # Scale data
        self.scaler_dict = {'MinMax': MinMaxScaler(),'Standard': StandardScaler()}
        # MinMaxScaler scales the data to be within a specific range, usually between 0 and 1
        # StandardScaler scales the data to have a mean of zero and a standard deviation of one
        self.scaler_type = scaler # Scaling type
        self.scale_by = 'slide' # Define how to scale. Available options: whole, patient, slide

        # Variables for machine learning step
        self.categorical_variables = ['Molecular.profile2', 'therapy_sequence','figo_stage']
        self.best_params = None
        self.RF = RandomForestClassifier(random_state = self.rng, class_weight = 'balanced')

        # Define the evaluation procedure
        self.cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=self.rng)
        
        # Save results 
        self.train_results = None
        self.train_time = None
        self.predictions_train = None
        
        self.test_results = {}
        self.predictions_test = None

        self.feature_selection_results = {}

    def choose_features(self):
        self.df = self.df.loc[:,self.features]
        logger.info(f"\nChose features successfully in {self.exp_name}.")
    
    def transform_df(self):

        try:
            if self.transformation == 'LOG':
                self.df.loc[:, self.features_to_transform] = np.log(self.df.loc[:, self.features_to_transform] + 1)

            elif self.transformation == 'LOG2':
                self.df.loc[:, self.features_to_transform] = np.log2(self.df.loc[:, self.features_to_transform] + 1)
                
            elif self.transformation == 'BOXCOX':
                # transform data & save lambda value
                for feature in self.features_to_transform:
                    self.df[feature], _ = stats.boxcox(self.df[feature].values)

            else:
                raise ValueError("Invalid transformation specified.")
            
            logger.info(f"\nTransformed DataFrame successfully in {self.exp_name}.")
            
        except Exception as e:
            logger.error(f"\nAn error occurred during {self.exp_name} data transformation: {e}", exc_info=True)

            
    def remove_outliers(self):
        
        try:
            if self.operation == 'trim_by_slide':
                slides = self.df['cycif.slide'].unique()
            
                for slide in slides:
                    # Create a mask to filter data belonging to the current slide
                    slide_mask = self.df['cycif.slide'] == slide
                    
                    for feature in self.features_for_outliers:
                        percentiles = np.percentile(self.df.loc[slide_mask, feature], [1, 99])
                        
                        # Replace values below the 1st percentile with the 1st percentile value
                        self.df.loc[slide_mask & (self.df[feature] < percentiles[0]), feature] = percentiles[0]
                        # Replace values above the 99th percentile with the 99th percentile value
                        self.df.loc[slide_mask & (self.df[feature] > percentiles[1]), feature] = percentiles[1]

            elif self.operation == 'remove':

                df_sub = self.df.loc[:, self.features_for_outliers]

                # Identify outliers using the 1st (0.01) and 99th (0.99) percentiles for each feature
                # For each data point, 'lim' will be True if the value is within the range [0.01, 0.99], otherwise False
                lim = np.logical_and(df_sub < df_sub.quantile(0.99, numeric_only=False),
                                df_sub > df_sub.quantile(0.01, numeric_only=False))

                # Data points outside the range [0.01, 0.99] will be replaced with NaN
                self.df.loc[:, self.features_for_outliers] = df_sub.where(lim, np.nan)
                
                # Drop rows with NaN in numerical columns
                self.df.dropna(subset=self.features_for_outliers, inplace=True)
            
            else: 
                raise ValueError("Invalid operation specified.")
            
            logger.info(f"\nRemoved outliers successfully in {self.exp_name}.")
        
        except Exception as e:
            logger.error(f"\nAn error occurred during {self.exp_name} outliers removal: {e}", exc_info=True)
    
    def scaler(self, df):
        
        # Get a scaler from the dictionary of supported scaler types
        scaler = self.scaler_dict.get(self.scaler_type)
        
        # Scale the data according to the specified scaler type
        data = scaler.fit_transform(df)
        
        return data
    
    def scaling(self):
        
        try:
            if self.scaler_type in self.scaler_dict:
                
                if self.scale_by not in ['whole','patient', 'slide']:
                    raise ValueError(f"Invalid order: {self.scale_by}")
                
                if self.scale_by == 'patient':
                    patients = self.df['patient'].unique()
                    # Iterate through each unique patient ID and scale the specified features for each patient separately
                    for patient in patients:
                        self.df.loc[self.df['patient'] == patient, self.features_to_scale] = self.scaler(self.df.loc[self.df['patient'] == patient, self.features_to_scale])
                
                elif self.scale_by == 'slide':
                    slides = self.df['cycif.slide'].unique()
                    # Iterate through each unique slide and scale the specified features for each slide separately
                    for slide in slides:
                        self.df.loc[self.df['cycif.slide'] == slide, self.features_to_scale] = self.scaler(self.df.loc[self.df['cycif.slide'] == slide, self.features_to_scale])
                
                elif self.scale_by == 'whole':
                    # Scale the specified features on the entire DataFrame
                    self.df.loc[self.features_to_scale] = self.scaler(self.df.loc[self.features_to_scale])
                
                logger.info(f"\nScalled successfully in {self.exp_name}.")
        
        except Exception as e:
            logger.error(f"\nAn error occurred during {self.exp_name} scaling: {e}", exc_info=True)

    def choose_types_of_cells(self):

        # Replace all immune cells as Immune
        self.df["GlobalCellType"] = self.df["GlobalCellType"].replace(to_replace = ['CD8.T.cells', 'CD4.T.cells'], value = 'Immune')
        
        cell_types = self.df["GlobalCellType"].unique()

        unique_patients = self.df['patient'].unique()
        
        # Remove patients with too few cells for each chosen cell type
        for cell_type in cell_types:
            # Group df by patient and get the size of each group
            grouped_df = self.df[self.df['GlobalCellType'] == cell_type].groupby('patient').size()

            # Filter patients with 100 or fewer cells and store them
            selected_patients = grouped_df[grouped_df <= 100].index

            # Remove patients with 100 and fewer cells
            if len(selected_patients) > 0:
                self.df = self.df[~self.df['patient'].isin(selected_patients)]

        unique_patients = self.df['patient'].unique()
        logger.info(f"\nAfter thresholding number of patients: {len(unique_patients)} in {self.exp_name}")

        try:
            # Leave only chosen cell types in df
            if self.types_of_cells == "Non-cancer":
                self.df = self.df[~self.df["GlobalCellType"].isin(['Cancer'])]

            elif self.types_of_cells in ["Cancer", "Stromal", "Immune"]:
                self.df = self.df[self.df["GlobalCellType"] == self.types_of_cells]

            elif self.types_of_cells not in ["All"]:
                raise ValueError("Invalid cell type(s) specified.")
        except Exception as e:
            logger.error(f"\nAn error occurred in {self.exp_name} while choosing cell type: {e}", exc_info=True)
        
        logger.info(f"\nChose successfully cell types in {self.exp_name}: {self.df['GlobalCellType'].unique()}.")
        
        # Remove column GlobalCellType
        self.df = self.df.drop(columns='GlobalCellType')
        
    def choose_mol_profiles(self):
        
        if self.mol_profiles != 'All':
            self.df = self.df[self.df['Molecular.profile2'].isin(self.mol_profiles)]

        logger.info(f"\nChose successfully classes of molecular profiles in {self.exp_name}: {self.df['Molecular.profile2'].unique()}.")

    def choose_therapy_sequences(self):
        if self.therapies != 'All':
            self.df = self.df[self.df['therapy_sequence'] == self.therapies]
        
        logger.info(f"\nChose successfully {self.therapies} therapy sequence(s) in {self.exp_name}: {self.df['therapy_sequence'].unique()}.")

    def prepare_categorical_inputs(self):

        # Encode categorical variables 
        self.label_encoder_dict = {}
        for variable in self.categorical_variables:
            
            label_encoder = LabelEncoder()
            
            # Convert the values of the current categorical variable to strings and encode them
            # The encoded values will replace the original values in the DataFrame
            self.df[variable] = label_encoder.fit_transform(self.df[variable].astype('str'))
            
            # Save the encoding results for the current variable in 'label_encoder_dict'
            # The dictionary will store the unique class labels and their corresponding encoded values
            self.label_encoder_dict[variable] = {
                'label_codes': label_encoder.classes_.tolist(), 
                'label_values': label_encoder.transform(label_encoder.classes_).tolist()
            }
        
        logger.info(f"\nPrepared successfully categorical inputs in {self.exp_name}.")

    def prep_train_test_data(self):
    
        # To receive same number of patients independent of runs
        np.random.seed(self.rng)
        
        # Group df by Molecular profile and therapy
        grouped = self.df.groupby(['Molecular.profile2', 'therapy_sequence'])

        # Assign 80% of patients from each unique group of Molecular profile and therapy to training and rest to test sets
        self.X_train_full = grouped.apply(lambda x: x.loc[x['patient'].isin(np.random.choice(x['patient'].unique(), size=int(0.8*len(x['patient'].unique())), replace=False))])
        self.X_test_full = grouped.apply(lambda x: x.loc[x['patient'].isin(np.setdiff1d(x['patient'].unique(), self.X_train_full['patient']))])
        
        # Assign target columns to y_train and y_test
        self.y_train=self.X_train_full[self.target]
        self.y_test=self.X_test_full[self.target]
        
        # Drop target columns
        self.X_train = self.X_train_full.drop(columns = [self.target, 'patient', 'cycif.slide','Molecular.profile2'])
        self.X_test = self.X_test_full.drop(columns = [self.target, 'patient', 'cycif.slide','Molecular.profile2'])
        
        logger.info(f"\nPrepared X_train of size {self.X_train.shape}, y_train of size {len(self.y_train)} and X_test of size {self.X_test.shape}, y_test of size {len(self.y_test)} in experiment {self.exp_name}.")

    def find_RF_by_Bayesian(self, X_train, y_train):
        '''
        inputs:
           - X_train: features training set
           - y_train: classes' labels of training set
        '''
        try:
            # Define hyperparameters search space 
            space = [Integer(200, 900, name='min_samples_split'),
                    Integer(200, 900, name='min_samples_leaf'),
                    Integer(50, 200, name='n_estimators')]
            
            # Store standard deviations of cross-validated scores
            std_devs = []
        
            # Define evaluation for each call within hyperparameters' space
            @use_named_args(space)
            def evaluate_model(**params):
                scores = cross_val_score(self.RF, X_train, y_train, scoring="f1_weighted", cv=self.cv, n_jobs = -1)
                std_devs.append(np.std(scores))
                return -np.mean(scores)
            
            # Perform Bayesian search on defined hyperparameters search space 
            start = time() # Get start time
            self.train_results = gp_minimize(evaluate_model, space, random_state=self.rng, n_random_starts=5, n_calls = 30)
            end = time() # Get end time
            
            # Calculate the training time
            self.train_time = end - start

            logger.info(f"\nAverage standard deviation of cross-validated scores in {self.exp_name}: {np.mean(std_devs)}")
            logger.info(f"\nTraining_time in {self.exp_name}: {self.train_time}")
            
            # Store the best hyperparameters
            self.best_params = {'min_samples_split': self.train_results.x[0],
                        'min_samples_leaf': self.train_results.x[1],
                        'n_estimators': self.train_results.x[2]}
            
            # Create RF model with stored best hyperparameters
            self.RF = RandomForestClassifier(**self.best_params, random_state = self.rng, class_weight = 'balanced')  

            logger.info(f"\nSuccessfully created RF model in {self.exp_name} with hyperparameters: {self.train_results.x} ( score - {-self.train_results.fun:.5f})")
        
        except Exception as e:
            logger.error(f"\nAn error occurred in find_RF_by_Bayesian during {self.exp_name}: {e}", exc_info=True)
        
    def test(self, X_train, y_train, X_test_full, X_test, y_test):
        '''
        inputs:
           - X_train: features training set
           - y_train: classes' labels of training set
           - X_test_full: features testing set with saved patient and slide IDs
           - X_test: features testing set
           - y_test: classes' labels of testing set
        '''
        
        # Training step: Fit the learner to the training data
        self.RF.fit(X_train, y_train)
            
        # Get the predictions on the training set(X_train),
        # then get predictions on the test samples(X_test)
        self.predictions_train = self.RF.predict(X_train)
        # Test step:
        self.predictions_test = self.RF.predict(X_test)
        
        # Save the predictions of X_test as df:
        # 1. Convert predictions_test to a pandas series
        predictions_series = pd.Series(self.predictions_test, name='predicted_label')
        # 2. Convert MultiIndex to a flat index in X_test
        X_test_flat = X_test_full.reset_index(drop=True)
        # 3. Concatenate X_test_flat and predictions_series along axis=1 to receive df of test set predictions 
        self.predictions_df = pd.concat([X_test_flat, predictions_series], axis=1)
                
        # Compute balanced accuracy on training samples
        self.balanced_accuracy_score_train = balanced_accuracy_score(y_train, self.predictions_train)
            
        # Compute balanced accuracy on the test set
        self.balanced_accuracy_score_test = balanced_accuracy_score(y_test, self.predictions_test)
        
        # Compute weighted F1-score on training samples
        self.f1_score_train = f1_score(y_train, self.predictions_train, average='weighted')
            
        # Compute weighted F1-score on the test set
        self.f1_score_test = f1_score(y_test, self.predictions_test, average='weighted')
        
        logger.info(f"\nSuccessfully tested RF model in {self.exp_name}")
    
    def calculate_mean_per_patient(self):
        name = self.exp_name + '_patients_predictions.csv'
        patients_predictions = self.predictions_df.groupby('patient')['predicted_label'].mean().reset_index()
        patients_predictions.to_csv(name, index=False)
        
        logger.info(f"\nSuccessfully calculated patients' predictions of RF model in {self.exp_name}")

    def eliminate_features(self, X_train, y_train):
        '''
        inputs:
        - X_train: features training set
        - y_train: classes' labels of training set
        - save_statement: define if you want to save figures 
        '''
        try:
            # Define cross-validated recursive feature elimination 
            # RFECV automatically selects the best number of features based on the provided scoring function of accuracy
            rfecv = RFECV(
                estimator=self.RF,
                min_features_to_select=5,
                step=5,
                n_jobs=-1,
                scoring="accuracy",
                cv=self.cv,
            )

            # Run cross-validated recursive feature elimination to select the most predictive features
            rfecv.fit(X_train.values, y_train.values)
            
            # Get the support mask indicating which features were selected by RFECV, where True values in the mask correspond to important features.
            support = rfecv.support_

            # Get the list of most predictive and eliminated features
            features = X_train.columns
            self.eliminated_features = list(features[~support])
            self.important_features = features[support]

            logger.info(f"\nImportant features in {self.exp_name}: {self.important_features}")
            logger.info(f"\nEliminated features in {self.exp_name}:  {self.eliminated_features}")

            # Calculate feature importances using mean decrease in impurity (Gini impurity)
            importances = rfecv.estimator_.feature_importances_

            self.X_train_transformed = X_train[self.important_features]

            # Calculate permutation importances
            permutation_scores = permutation_importance(rfecv.estimator_, self.X_train_transformed.values, y_train.values, n_repeats=10, random_state=self.rng, n_jobs=-1)

            feature_permutations = dict(zip(self.important_features, permutation_scores.importances_mean))
            self.permutation_scores_dict = dict(sorted(feature_permutations.items(), key=lambda x: x[1], reverse=True))

            # For gini impurity:
            # Create dictionary with important feature names as keys and importances as values
            feature_importances = dict(zip(self.important_features, importances))
            self.importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
        
        except Exception as e:
            logger.error(f"\nAn error occurred during {self.exp_name} in eliminate_features: {e}", exc_info=True)

    def save_values(self):

        # Compute balanced accuracy on training samples
        self.test_results['balanced_acc_train'] = self.balanced_accuracy_score_train
            
        # Compute balanced accuracy on the test set
        self.test_results['balanced_acc_test'] = self.balanced_accuracy_score_test
        
        # Compute weighted F1-score on training samples
        self.test_results['f1_train'] = self.f1_score_train

        # Compute weighted F1-score on the test set
        self.test_results['f1_test'] = self.f1_score_test

        self.feature_selection_results['importances'] = self.importances
        self.feature_selection_results['eliminated_features'] = self.eliminated_features

        self.feature_selection_results['permutation_scores'] = self.permutation_scores_dict

def prep_df():

    all_cells = pd.read_csv('All_cells_subtype-info_20231028.csv', index_col=False)
    all_cells['CellId'] = all_cells['cycif.slide'] + '_' + all_cells['cycif.core.id'] + '_c' + all_cells['CellId'].astype(str)
    
    # Load spatial proportions
    spatial_proportions = pd.read_csv("Spatial-cell-counts_radious46px_20231101.csv")

    # Merge cancer cells and their corresponding spatial proportions on CellId
    dataset = pd.merge(all_cells, spatial_proportions, on="CellId")
    
    # Load the text file into a pandas DataFrame
    slides_explanations = pd.read_csv('All-slides_cycif2samples.txt', sep="	")

    # Load molecular profile 2
    molecular_profiles = pd.read_csv("Molecular_profiles_patients_20231123.csv")
    
    # Load patinets' clinical info
    patients_data = pd.read_csv('33_66_percentiles_Clinical_outcome_T-CycIF.csv')
    patients_data.rename(columns = {'TnumberAVL':'patient'}, inplace = True)
    
    final_patients_data = pd.merge(molecular_profiles, patients_data[["patient","classes_of_timelastfu","figo_stage","refage"]], on=["patient"])

    # Merge the two data frames on the slide number and core id columns
    combined_dataset = pd.merge(dataset, slides_explanations[["cycif.slide", "cycif.core.id","patient"]], on=["cycif.slide", "cycif.core.id"])

    # Merge the two data frames on patient column
    df = pd.merge(combined_dataset, final_patients_data, on=["patient"])
    
    # print("df's columns: ", df.columns)

    df['therapy_sequence'] = df['therapy_sequence'].replace(to_replace='NACT', value='IDS', regex=True)
    df['therapy_sequence'] = df['therapy_sequence'].replace(to_replace='IDS followed by re-debulking', value='IDS', regex=True)
    df['therapy_sequence'] = df['therapy_sequence'].replace(to_replace='PDS followed by IDS', value='IDS', regex=True)
    df['therapy_sequence'] = df['therapy_sequence'].replace(['Primairy debulking', 'Only debulking'], 'PDS')
    
    df = df.dropna()

    return df

def perform_ml_run(exp_name, channels, channels_to_transform, channels_to_outliers, channels_to_scale, cells, target, classes, therapies, scaler, best_params, acc_train, acc_test, f_train, f_test, pred_features, eliminated_features, permutation_scores, random_seed, name, exps, df):

    model = Model(exp_name, channels, channels_to_transform, channels_to_outliers, channels_to_scale, cells, target, classes, therapies, scaler, df)

    model.choose_features()
    model.transform_df()
    model.remove_outliers()
    model.scaling()
    model.choose_types_of_cells()
    model.choose_mol_profiles()
    model.choose_therapy_sequences()

    model.prepare_categorical_inputs()
    model.prep_train_test_data()
    model.find_RF_by_Bayesian(model.X_train.values, model.y_train.values)
    model.test(model.X_train.values, model.y_train.values, model.X_test_full, model.X_test.values, model.y_test.values)
    model.calculate_mean_per_patient()
    model.eliminate_features(model.X_train, model.y_train)
    model.save_values()

    # Collect the results in a dictionary
    results = {
        'exp_name': exp_name,
        'best_parameters': model.best_params,
        'balanced_acc_train': model.test_results['balanced_acc_train'],
        'balanced_acc_test': model.test_results['balanced_acc_test'],
        'f1_train': model.test_results['f1_train'],
        'f1_test': model.test_results['f1_test'],
        'most_predictive_features': model.feature_selection_results['importances'],
        'eliminated_features': model.feature_selection_results['eliminated_features'],
        'permutation_scores': model.feature_selection_results['permutation_scores']
    }

    logger.info(f"\n{exp_name} is done") 

    return results

def run_ml_experiments(data):
    exp_name, channels, channels_to_transform, channels_to_outliers, channels_to_scale, types_of_cells, classes_column, \
    therapies, mol_profiles, scaling_type, best_parameters, balanced_acc_train, balanced_acc_test, f1_train, f1_test, \
    most_predictive_features, eliminated_features, permutation_scores, random_seed, name, exps, df = data

    # Call the perform_ml_run function with the provided parameters
    result = perform_ml_run(exp_name, channels, channels_to_transform, channels_to_outliers, channels_to_scale,
                             types_of_cells, classes_column, therapies, mol_profiles, scaling_type, best_parameters,
                             balanced_acc_train, balanced_acc_test, f1_train, f1_test, most_predictive_features,
                             eliminated_features, permutation_scores, random_seed, name, exps, df)

    return result

if __name__ == '__main__':

    try:
        df = pd.read_csv("DATASET_FILENAME.csv")
        df['Molecular.profile2'] = df['Molecular.profile2'].replace('BRCAmut/met', 'BRCAmutmet')
        
    except Exception as e:
        logger.error(f"\nAn error occurred while reading dataset file: {e}", exc_info=True) 
    
    # Get the number of available CPU cores for parallel processing
    max_workers = mp.cpu_count()
    
    # Specify your exps files
    names = ['EXPERIMENTS_FILE_NAME.csv', ...]

    for name in names:
        
        try:
            exps = pd.read_csv(name)
        except Exception as e:
            logger.error(f"\nAn error occurred while reading experiments' file: {e}", exc_info=True) 

        # Use loky's get_reusable_executor for parallel execution of multiple experiments
        with get_reusable_executor(max_workers=max_workers, timeout=2) as executor:
            data_list = [(d['experiment'], d['channels'], d['channels_to_transform'], d['channels_to_outliers'], d['channels_to_scale'], d['types_of_cells'], d['classes_column'], d['therapies'], d['mol_profiles'], d['scaling_type'], d['best_parameters'], d['balanced_acc_train'], d['balanced_acc_test'], d['f1_train'], d['f1_test'], d['most_predictive_features'], d['eliminated_features'], d['permutation_scores'], d['random_seed'], name, exps, df) for i, d in exps.iterrows()]
            
            # Perform multiple machine learning runs in parallel using executor.map()
            results = list(executor.map(run_ml_experiments, data_list))

        # Update the experiment DataFrame with the results obtained from the machine learning runs
        for result in results:
            exp_name = result['exp_name']
            exps.loc[exps['experiment'] == exp_name, 'best_parameters'] = str(result['best_parameters'])
            exps.loc[exps['experiment'] == exp_name, 'balanced_acc_train'] = result['balanced_acc_train']
            exps.loc[exps['experiment'] == exp_name, 'balanced_acc_test'] = result['balanced_acc_test']
            exps.loc[exps['experiment'] == exp_name, 'f1_train'] = result['f1_train']
            exps.loc[exps['experiment'] == exp_name, 'f1_test'] = result['f1_test']
            exps.loc[exps['experiment'] == exp_name, 'most_predictive_features'] = str(result['most_predictive_features'])
            exps.loc[exps['experiment'] == exp_name, 'eliminated_features'] = str(result['eliminated_features'])
            exps.loc[exps['experiment'] == exp_name, 'permutation_scores'] = str(result['permutation_scores'])

        # Save the updated DataFrame to experiments file
        exps.to_csv(name, index=False)
        logger.info("\nStored successfully experiments' file")