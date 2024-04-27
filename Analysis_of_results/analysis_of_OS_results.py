import os
import ast

import numpy as np
import pandas as pd 
from collections import defaultdict

from scipy import stats
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

# Set the font scale and style
sns.set(style="ticks", palette="pastel", rc={'figure.facecolor': (0, 0, 0, 0)})

exps = {
    'Cancer1': 'TIM3,pSTAT1,CD45RO,CD20,CD11c,CD207,GranzymeB,CD163,CD4,CD3d,CD8a,FOXP3,PD1,CD15,PDL1_488,Ki67,Vimentin,MHCII,MHCI,ECadherin,aSMA,CD31,pTBK1,CK7,yH2AX,cPARP1,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune,figo_stage,refage',
    'Cancer2': 'pSTAT1,pTBK1,yH2AX,cPARP1,PDL1_488,Ki67,Vimentin,CK7,MHCI,ECadherin,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune,figo_stage,refage',
    'Non-cancer': 'TIM3,pSTAT1,CD45RO,CD20,CD11c,CD207,GranzymeB,CD163,CD4,CD3d,CD8a,FOXP3,PD1,CD15,PDL1_488,Ki67,Vimentin,MHCII,MHCI,ECadherin,aSMA,CD31,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune,figo_stage,refage',
    'All': 'TIM3,pSTAT1,CD45RO,CD20,CD11c,CD207,GranzymeB,CD163,CD4,CD3d,CD8a,FOXP3,PD1,CD15,PDL1_488,Ki67,Vimentin,MHCII,MHCI,ECadherin,aSMA,CD31,pTBK1,CK7,yH2AX,cPARP1,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune,figo_stage,refage'
}

def calculate_median_f1_score(files, celltype = 'Other'):
    
    if celltype == 'Cancer':
        combined_datasets = {}
        for file in files:
            base_filename = file.rsplit(".", 1)[0]
            combined_data = pd.read_csv(file)
            combined_datasets[base_filename] = combined_data
    
    else:
        # Create a dictionary to group files by base filename
        file_groups = {}
        for file in files:
            base_filename = '_'.join(file.split('_')[:-1])  # Exclude the last part (number) from the filename
            if base_filename not in file_groups:
                file_groups[base_filename] = []
            file_groups[base_filename].append(file)
        
        # Combine datasets for each base filename
        combined_datasets = {}
        for base_filename, file_list in file_groups.items():
            combined_data = pd.concat([pd.read_csv(file) for file in file_list], axis=0)
            combined_datasets[base_filename] = combined_data
        
    
    # Create a dictionary to store median values for each dataset
    median_results = {}

    # Compute median for each dataset
    for base_filename, combined_data in combined_datasets.items():
        
        median_values = combined_data['f1_test'].median()
        
        # Calculate the confidence interval
        n = len(combined_data)
        df = n - 1  # degrees of freedom
        standard_error = combined_data['f1_test'].std() / np.sqrt(n)
        margin_of_error = stats.t.ppf(0.975, df) * standard_error
        ci_lower = median_values - margin_of_error
        ci_upper = median_values + margin_of_error

        median_results[base_filename] = {
            'median': median_values,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    return median_results

def create_heatmap_of_f1(all_results,cancer1_results,cancer2_results,non_cancer_results,therapy):

    # Get the union of all class pairs
    all_class_pairs = set(cancer1_results.keys()).union(set(cancer2_results.keys()), set(non_cancer_results.keys()), set(all_results.keys()))

    # Combine data into a single DataFrame with NaN for missing class pairs
    combined_data = pd.DataFrame({
        'All.cells': [all_results.get(pair, {'median': None, 'confidence_interval': (None, None)})['median'] for pair in all_class_pairs],
        'Cancer.full.panel': [cancer1_results.get(pair, {'median': None, 'confidence_interval': (None, None)})['median'] for pair in all_class_pairs],
        'Cancer': [cancer2_results.get(pair, {'median': None, 'confidence_interval': (None, None)})['median'] for pair in all_class_pairs],
        'Immune-stromal': [non_cancer_results.get(pair, {'median': None, 'confidence_interval': (None, None)})['median'] for pair in all_class_pairs],
    }, index=list(all_class_pairs))

    # Extract class pairs from the index
    combined_data['ClassPair'] = combined_data.index.str.split('_').str[1:-1].str.join('_')

    # Reorganize the DataFrame
    combined_data = combined_data.groupby('ClassPair').mean()
        
    # Create a heatmap
    plt.figure(figsize=(combined_data.shape[1],combined_data.shape[0]))

    heatmap = sns.heatmap(combined_data, annot=True, cmap=sns.cubehelix_palette(start=2.5, rot=0, dark=0.2, light=0.97, reverse=False, as_cmap=True), fmt='.2f', linewidths=.5, vmin=0.44, vmax=0.8, cbar_kws={'label': 'F1 test Median Value'})

    # Add confidence intervals as annotations
    for i, row in enumerate(combined_data.iterrows()):
        _, data = row
        for j, (col, value) in enumerate(data.iteritems()):
            if pd.notna(value):
                # Choose the correct dataset based on the column
                if col == 'All.cells':
                    ci_lower, ci_upper = all_results['All_{}_{}'.format(data.name, therapy)]['confidence_interval']
                elif col == 'Cancer.full.panel':
                    ci_lower, ci_upper = cancer1_results['Cancer_{}_{}'.format(data.name, therapy)]['confidence_interval']
                elif col == 'Cancer':
                    ci_lower, ci_upper = cancer2_results['Cancer_{}_{}'.format(data.name, therapy)]['confidence_interval']
                elif col == 'Immune-stromal':
                    ci_lower, ci_upper = non_cancer_results['Immune-stromal_{}_{}'.format(data.name, therapy)]['confidence_interval']
                
                text_color = 'white' if value > 0.59 else 'black'

                # Add confidence interval in brackets below the median value
                plt.text(j + 0.5, i + 0.7, f"({ci_lower:.2f}, {ci_upper:.2f})", ha='center', va='center', color=text_color, fontsize=6)
    
    plt.ylabel("") 
    custom_yticks = ['Low or high in HRDs', 'Low or high in HRPs']  
    heatmap.set_yticklabels(custom_yticks, rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tick_params(left = False, bottom = False)
        
    plt.title(therapy)
    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/OS/preds_heatmap_{therapy}.svg")
    plt.show()
    
def prep_df(df):
    
    features = ['TIM3','pSTAT1','CD45RO','CD20','CD11c','CD207','GranzymeB','CD163','CD4','CD3d','CD8a','FOXP3','PD1','CD15','PDL1_488','Ki67','Vimentin','MHCII','MHCI','ECadherin','aSMA','CD31','pTBK1','CK7','yH2AX','cPARP1', 'Cancer', 'CD4.T.cells','Area','Eccentricity','Roundness','GlobalCellType','Molecular.profile2', 'therapy_sequence','cycif.slide','classes_of_timelastfu','refage','figo_stage']
    df = df[features]

    # Define markers to transform
    markers = ['TIM3','pSTAT1','CD45RO','CD20','CD11c','CD207','GranzymeB','CD163','CD4','CD3d','CD8a','FOXP3','PD1','CD15','PDL1_488','Ki67','Vimentin','MHCII','MHCI','ECadherin','aSMA','CD31','pTBK1','CK7','yH2AX','cPARP1']

    ### 1. Log2 transform dataset on selected features 
    df.loc[:, markers] = df.loc[:, markers] = np.log2(df.loc[:, markers] + 1)

    ### 2. Remove outliers 

    channels = list(markers)
    channels.extend(['Area','Eccentricity','Roundness'])

    df_sub = df.loc[:, channels]

    # Identify outliers using the 1st (0.01) and 99th (0.99) percentiles for each feature
    # For each data point, 'lim' will be True if the value is within the range [0.01, 0.99], otherwise False
    lim = np.logical_and(df_sub < df_sub.quantile(0.99, numeric_only=False),
                    df_sub > df_sub.quantile(0.01, numeric_only=False))

    # Data points outside the range [0.01, 0.99] will be replaced with NaN
    df.loc[:, channels] = df_sub.where(lim, np.nan)

    # Drop rows with NaN in numerical columns
    df.dropna(subset=channels, inplace=True)

    ### 3. Scale

    # Get a scaler
    scaler = StandardScaler()

    slides = df['cycif.slide'].unique()
    # Iterate through each unique slide and scale the specified features for each slide separately
    for slide in slides:
        df.loc[df['cycif.slide'] == slide, channels] = scaler.fit_transform(df.loc[df['cycif.slide'] == slide, channels])

    return df
    
# Top 5 most predictive features
def plot_top_pred_features(df, column, number_of_features, return_values = False):
    # Initialize a defaultdict to store importance values for each feature
    feature_importance_dict = defaultdict(list)

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        features = ast.literal_eval(row[column])  # Convert string to dictionary
        for feature, importance in features.items():
            feature_importance_dict[feature].append(importance)

    # Sort features based on average importance values in descending order
    sorted_features = sorted(feature_importance_dict.keys(), key=lambda x: np.mean(feature_importance_dict[x]), reverse=True)
    
    # Take the top features
    top_features = sorted_features[:number_of_features]

    # Ensure all lists have the same length by padding with np.nan
    max_length = max(len(feature_importance_dict[feature]) for feature in top_features)
    for feature in top_features:
        feature_importance_dict[feature] += [np.nan] * (max_length - len(feature_importance_dict[feature]))

    # Create a DataFrame containing only the top 5 features and their values
    data = pd.DataFrame({feature: feature_importance_dict[feature] for feature in top_features})

    return data.median().to_dict() if return_values == True else top_features

def create_spider_plot_for_four_OS(cells_df, cell_type, therapy, top_markers):

    # Subset the cells df based on therapy, molecular profile, OS and cell type
    df = cells_df[cells_df["therapy_sequence"] == therapy]
    
    mol_profiles_df1 = df[df['Molecular.profile2'].isin(['BRCAmutmet','HRD'])]
    mol_profiles_df2 = df[df['Molecular.profile2'].isin(['CCNE1amp','HRP'])]
        
    mol_profiles1_df0 = mol_profiles_df1[mol_profiles_df1['classes_of_timelastfu'] == 0]
    mol_profiles1_df1 = mol_profiles_df1[mol_profiles_df1['classes_of_timelastfu'] == 1]
    
    mol_profiles2_df0 = mol_profiles_df2[mol_profiles_df2['classes_of_timelastfu'] == 0]
    mol_profiles2_df1 = mol_profiles_df2[mol_profiles_df2['classes_of_timelastfu'] == 1]

    if cell_type == "Non-cancer":
        celltype_df1_0 = mol_profiles1_df0[~mol_profiles1_df0["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df1_1 = mol_profiles1_df1[~mol_profiles1_df1["GlobalCellType"].isin(['Others', 'Cancer'])]
        
        celltype_df2_0 = mol_profiles2_df0[~mol_profiles2_df0["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df2_1 = mol_profiles2_df1[~mol_profiles2_df1["GlobalCellType"].isin(['Others', 'Cancer'])]

    elif cell_type in ["Cancer", "Stromal"]:
        celltype_df1_0 = mol_profiles1_df0[mol_profiles1_df0["GlobalCellType"] == cell_type]
        celltype_df1_1 = mol_profiles1_df1[mol_profiles1_df1["GlobalCellType"] == cell_type]
        
        celltype_df2_0 = mol_profiles2_df0[mol_profiles2_df0["GlobalCellType"] == cell_type]
        celltype_df2_1 = mol_profiles2_df1[mol_profiles2_df1["GlobalCellType"] == cell_type]
        
    elif cell_type == "Immune":
        celltype_df1_0 = mol_profiles1_df0[~mol_profiles1_df0["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df1_1 = mol_profiles1_df1[~mol_profiles1_df1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        
        celltype_df2_0 = mol_profiles2_df0[~mol_profiles2_df0["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df2_1 = mol_profiles2_df1[~mol_profiles2_df1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        
    elif cell_type == 'All':
        celltype_df1_0 = mol_profiles1_df0[~mol_profiles1_df0["GlobalCellType"].isin(['Others'])]
        celltype_df1_1 = mol_profiles1_df1[~mol_profiles1_df1["GlobalCellType"].isin(['Others'])]
        
        celltype_df2_0 = mol_profiles2_df0[~mol_profiles2_df0["GlobalCellType"].isin(['Others'])]
        celltype_df2_1 = mol_profiles2_df1[~mol_profiles2_df1["GlobalCellType"].isin(['Others'])]

    # Create a dictionary to record median values across OS groups
    markers_dict = {'OS': ['Low in HRDs','High in HRDs','Low in HRPs','High in HRPs']}
    
    for marker in top_markers:
        median1 = np.median(celltype_df1_0[marker].values)
        median2 = np.median(celltype_df1_1[marker].values)
        
        median3 = np.median(celltype_df2_0[marker].values)
        median4 = np.median(celltype_df2_1[marker].values)

        markers_dict[marker] = [median1,median2,median3,median4]
    
    # Convert dictionary to DataFrame and prepare it for plotting
    mean_df = pd.DataFrame(markers_dict)    
    melted_df = pd.melt(mean_df, id_vars=['OS'], var_name='theta', value_name='r')
    
    # Plot the median values across OS groups
    fig = px.line_polar(melted_df, r='r', theta='theta', color = 'OS', 
                        line_close=True, color_discrete_sequence = ['#e8b7bb','#D33F49','#A7AEF2','#343c82'])

    # Update the layout to increase text size
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {'size': 20},  # Set the font size to your preferred value
        'polar': {'radialaxis': {'range': [min(melted_df['r']), max(melted_df['r'])]}}
    })
    
    fig.write_image(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/OS/{cell_type}_Short_Long_in_HRDs_HRPs_{therapy}_{top_markers[0]}.svg")
    fig.show()

def create_violin_plot(cells_df, mol_profiles, cell_type, therapy, top_markers):
    
    # Update BRCAloss label BRCAmutmet present in initial cell dataframe
    if 'BRCAloss' in mol_profiles:
        index = mol_profiles.index('BRCAloss')
        mol_profiles[index] = 'BRCAmutmet'

    # Subset the cells df based on therapy, molecular profile, OS and cell type
    df = cells_df[cells_df["therapy_sequence"] == therapy]
    mol_profiles_df = df[df['Molecular.profile2'].isin(mol_profiles)]
        
    mol_profiles_df0 = mol_profiles_df[mol_profiles_df['classes_of_timelastfu'] == 0]
    mol_profiles_df1 = mol_profiles_df[mol_profiles_df['classes_of_timelastfu'] == 1]

    if cell_type == "Non-cancer":
        celltype_df1 = mol_profiles_df0[~mol_profiles_df0["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df2 = mol_profiles_df1[~mol_profiles_df1["GlobalCellType"].isin(['Others', 'Cancer'])]

    elif cell_type in ["Cancer", "Stromal"]:
        celltype_df1 = mol_profiles_df0[mol_profiles_df0["GlobalCellType"] == cell_type]
        celltype_df2 = mol_profiles_df1[mol_profiles_df1["GlobalCellType"] == cell_type]
        
    elif cell_type == "Immune":
        celltype_df1 = mol_profiles_df0[~mol_profiles_df0["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df2 = mol_profiles_df1[~mol_profiles_df1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        
    elif cell_type == 'All':
        celltype_df1 = mol_profiles_df0[~mol_profiles_df0["GlobalCellType"].isin(['Others'])]
        celltype_df2 = mol_profiles_df1[~mol_profiles_df1["GlobalCellType"].isin(['Others'])]

    # Concatenate data for all markers
    all_markers_data = pd.concat([celltype_df1[top_markers].stack(), celltype_df2[top_markers].stack()], axis=1)
    all_markers_data.columns = ['Short', 'Long']
    all_markers_data['Marker'] = all_markers_data.index.get_level_values(1)

    # Reset index to ensure 'Marker' column is a regular column
    all_markers_data.reset_index(drop=True, inplace=True)

    # Melt the dataframe to long format
    all_markers_data = all_markers_data.melt(id_vars=['Marker'], value_vars=['Short', 'Long'], var_name='OS', value_name='Pre-processed expression')

    # Define the figure arguments and test configuration
    fig_args = {'x': 'Marker',
            'y': 'Pre-processed expression',
            'hue':'OS',
            'data': all_markers_data,
            'order': top_markers,
            'hue_order':['Short', 'Long']}

    configuration = {'test':'Mann-Whitney',
                    'comparisons_correction':None,
                    'text_format':'star'}

    # Plot the distribution as violins
    ax = sns.violinplot(**fig_args, palette = {'Short': "#fed18c", 'Long': "#d2d2e6"}, fill=False)

    # Add the significance
    significanceComparisons = [
        ((marker, 'Short'), (marker, 'Long'))
        for marker in top_markers
    ]
    annotator = Annotator(ax, significanceComparisons, **fig_args, plot = 'violinplot')
    annotator.configure(**configuration).apply_test().annotate()
    
    # Save figure
    mol_profiles_names = "".join(mol_profiles)
    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/OS/{cell_type}_{mol_profiles_names}_Short_Long_{therapy}_{top_markers[0]}.svg")
    plt.show()

def create_heatmap_mean_value_per_marker(cells_df, therapy, cell_marker_df):
    
    # Pre-process raw cell dataframe
    cells_df = prep_df(cells_df)

    # Subset the cells df based on therapy, molecular profile, OS
    df = cells_df[cells_df["therapy_sequence"] == therapy]
    ccne1amp_hrp = df[df['Molecular.profile2'].isin(['CCNE1amp','HRP'])]
        
    ccne1amp_hrp0 = ccne1amp_hrp[ccne1amp_hrp['classes_of_timelastfu'] == 0]
    ccne1amp_hrp1 = ccne1amp_hrp[ccne1amp_hrp['classes_of_timelastfu'] == 1]
    
    # Drop the duplicates in markers dataframe
    cell_marker_df = cell_marker_df.drop_duplicates()
    
    # Remove rows with 'refage' and 'figo_stage' in Marker
    cell_marker_df = cell_marker_df[~cell_marker_df['Marker'].isin(['refage', 'figo_stage'])]
    
    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame(columns=['CellType', 'Marker', 'BRCAloss_HRD_poor', 'BRCAloss_HRD_high','CCNE1amp_HRP_poor', 'CCNE1amp_HRP_high', 'PvalueFromKruskal'])

    for index, row in cell_marker_df.iterrows():
        
        cell_type = row['CellType']
        marker = row['Marker']

        if cell_type == "Non-Cancer":
            celltype_ccne1amp_hrp0 = ccne1amp_hrp0[~ccne1amp_hrp0["GlobalCellType"].isin(['Others', 'Cancer'])]
            celltype_ccne1amp_hrp1 = ccne1amp_hrp1[~ccne1amp_hrp1["GlobalCellType"].isin(['Others', 'Cancer'])]
        elif cell_type in ["Cancer", "Stromal"]:
            celltype_ccne1amp_hrp0 = ccne1amp_hrp0[ccne1amp_hrp0["GlobalCellType"] == cell_type]
            celltype_ccne1amp_hrp1 = ccne1amp_hrp1[ccne1amp_hrp1["GlobalCellType"] == cell_type]
        elif cell_type == "Immune":
            celltype_ccne1amp_hrp0 = ccne1amp_hrp0[~ccne1amp_hrp0["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
            celltype_ccne1amp_hrp1 = ccne1amp_hrp1[~ccne1amp_hrp1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        elif cell_type == 'All':
            celltype_ccne1amp_hrp0 = ccne1amp_hrp0[~ccne1amp_hrp0["GlobalCellType"].isin(['Others'])]
            celltype_ccne1amp_hrp1 = ccne1amp_hrp1[~ccne1amp_hrp1["GlobalCellType"].isin(['Others'])]

        # Extract marker values for each group
        values_ccne1amp_hrp0 = celltype_ccne1amp_hrp0[marker].values
        values_ccne1amp_hrp1 = celltype_ccne1amp_hrp1[marker].values

        # Perform Kruskal-Wallis H-test
        h_stat, p_value = stats.kruskal(values_ccne1amp_hrp0, values_ccne1amp_hrp1) # , values_brcamutmet_hrd0, values_brcamutmet_hrd1
        print("Kruskal H-stat: ", h_stat, "P-value", p_value)

        # Add the results to the result_df
        result_df = result_df.append({'CellType': cell_type, 'Marker': marker,
                                    'CCNE1amp_HRP_poor': np.median(values_ccne1amp_hrp0), 'CCNE1amp_HRP_high': np.median(values_ccne1amp_hrp1),
                                    'PvalueFromKruskal': p_value},
                                    ignore_index=True)

    # Add a column indicating whether p-value is less than 0.05
    result_df['Significance'] = result_df['PvalueFromKruskal'] < 0.05
        
    # Combine CellType and Marker columns into a new column
    result_df['CellMarker'] = result_df['CellType'] + '_' + result_df['Marker']

    # Set the new column 'CellMarker' as the index
    result_df.set_index('CellMarker', inplace=True)
    
    # Create a heatmap with p-values
    plt.figure(figsize=(2, 5)) # =(4,(result_df.shape[0]-1))
    heatmap = sns.heatmap(result_df[['CCNE1amp_HRP_poor', 'CCNE1amp_HRP_high']], #, 'BRCAloss_HRD_poor', 'BRCAloss_HRD_high'
                        vmin=result_df[['CCNE1amp_HRP_poor', 'CCNE1amp_HRP_high']].values.min(), # , 'BRCAloss_HRD_poor', 'BRCAloss_HRD_high'
                        vmax=result_df[['CCNE1amp_HRP_poor', 'CCNE1amp_HRP_high']].values.max(), # , 'BRCAloss_HRD_poor', 'BRCAloss_HRD_high'
                annot=False, cmap=LinearSegmentedColormap.from_list('custom_cmap', ['#6599CD','#FFFFFF','#D33F49']), fmt='.2f', cbar_kws={'label': 'Median expression values'}) 
    
    # Add color indicators for significance
    for i, (cell_marker, row) in enumerate(result_df.iterrows()):
        if row['Significance']:
            plt.text(2, i + 0.6, "*", ha='center', va='center', color='black', fontsize=30) # (4,
    
    plt.xlabel("")
    custom_xticks = ['HRPs short', 'HRPs long']
    heatmap.set_xticklabels(custom_xticks, rotation=45, ha='right')
    
    plt.ylabel("Cell type & Marker")
    plt.title(therapy)
    plt.tick_params(left = False, bottom = False) 
    
    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/OS/predictive_features_{therapy}.svg") 
    plt.show()
    
def plot_oncogrid_of_pred_features(result_df,therapy):
    
    result_df = result_df.transpose()

    # Calculate the count of non-null values in each row
    non_null_counts = result_df.notnull().sum(axis=1)

    # Sort the DataFrame based on the count of non-null values in each row
    df_sorted = result_df.iloc[non_null_counts.argsort()[::-1]]

    # Function to create the mask
    def create_mask(celltype_df, exps):
        mask_df = pd.DataFrame(0, index=celltype_df.index, columns=celltype_df.columns)
        for col in celltype_df.columns:
            celltype = col.split('_')[0]
            markers = exps[celltype].split(',')
            for marker in celltype_df.index:
                mask_df.loc[marker, col] = marker not in markers
        return mask_df

    # Create the mask
    mask_df = create_mask(df_sorted, exps)

    # Function to remove the first item before '_'
    def remove_first_item_before_underscore(column_name):
        index = column_name.find('_')  # Find the index of the first '_'
        return column_name[index+1:] if index != -1 else column_name

    # Update column names in the DataFrame
    df_sorted.columns = [remove_first_item_before_underscore(col) for col in df_sorted.columns]
    mask_df.columns = [remove_first_item_before_underscore(col) for col in mask_df.columns]

    sns.set(style="ticks", palette="pastel", rc={'figure.facecolor': (0, 0, 0, 0)}, font_scale=1.8)
    xtick_labels = ['HRDs', 'HRPs',
                    'HRDs', 'HRPs',
                    'HRDs', 'HRPs',
                    'HRDs', 'HRPs']

    # Create heatmap
    plt.figure(figsize=(8, 8))

    # Overlay a second heatmap with a single color for True values in mask_df
    sns.heatmap(1-mask_df.astype(bool), cmap=LinearSegmentedColormap.from_list('custom_cmap', ['#e3e3e8','#ffffff']), cbar=False, annot=False) # Greys #d2d2e6

    heatmap = sns.heatmap(df_sorted, cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffefd8','#fdb340']),
                        annot=False, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Median MDA'})
    for i in range(1, 6):
        heatmap.axvline(i * 2, color='white', linewidth=5)

    heatmap.set_xticklabels(xtick_labels, rotation=45, ha='right')  # Set custom xtick labels
    plt.tick_params(left = False, bottom = False) 

    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/OS/oncogrid_of_predictive_features_{therapy}.svg")
    plt.show()
    
def process_results(chosen_results, folder_path, result_type, result_df, number_of_features, final_plot = False):
    
    results_dict = {}

    # Collect predictive features for corresponding comparison
    for chosen_comparison in chosen_results:

        cell_type, class1, class2, _ = chosen_comparison.split("_")

        os.chdir(folder_path)

        if result_type in ['Cancer','Cancer1','Cancer2']:
            df = pd.read_csv(f'{chosen_comparison}.csv')
        else:
            df = pd.concat([pd.read_csv(f'{chosen_comparison}_1.csv'), pd.read_csv(f'{chosen_comparison}_2.csv')], axis=0)

        if final_plot == 'oncogrid':
            median_values = plot_top_pred_features(df, 'permutation_scores', number_of_features=number_of_features, return_values=True)
            results_dict[f'{result_type}_{class1}_vs_{class2}'] = median_values

        elif final_plot == 'heatmap':
            top_features = plot_top_pred_features(df, 'permutation_scores', number_of_features)
            results_dict[chosen_comparison] = top_features

    # Prepare the features dataframe for plotting
    if final_plot == 'oncogrid':

        # Create DataFrame
        df = pd.DataFrame(results_dict).transpose()

        # Append results to final dataframe
        result_df = result_df.append(df)

        return result_df

    elif final_plot == 'heatmap':
        common_items = set(results_dict[next(iter(results_dict))])

        for value in results_dict.values():
            common_items = common_items.intersection(value)

        for common_item in common_items:
            result_row = {'CellType': cell_type, 'Marker': common_item}
            result_df = result_df.append(result_row, ignore_index=True)

    return result_df