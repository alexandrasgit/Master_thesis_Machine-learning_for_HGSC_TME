import os
import ast

import numpy as np
import pandas as pd 
from collections import defaultdict

from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.stats import mannwhitneyu
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
    'Cancer1': 'TIM3,pSTAT1,CD45RO,CD20,CD11c,CD207,GranzymeB,CD163,CD4,CD3d,CD8a,FOXP3,PD1,CD15,PDL1_488,Ki67,Vimentin,MHCII,MHCI,ECadherin,aSMA,CD31,pTBK1,CK7,yH2AX,cPARP1,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune',
    'Cancer2': 'pSTAT1,pTBK1,yH2AX,cPARP1,PDL1_488,Ki67,Vimentin,CK7,MHCI,ECadherin,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune',
    'Non-cancer': 'TIM3,pSTAT1,CD45RO,CD20,CD11c,CD207,GranzymeB,CD163,CD4,CD3d,CD8a,FOXP3,PD1,CD15,PDL1_488,Ki67,Vimentin,MHCII,MHCI,ECadherin,aSMA,CD31,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune',
    'All': 'TIM3,pSTAT1,CD45RO,CD20,CD11c,CD207,GranzymeB,CD163,CD4,CD3d,CD8a,FOXP3,PD1,CD15,PDL1_488,Ki67,Vimentin,MHCII,MHCI,ECadherin,aSMA,CD31,pTBK1,CK7,yH2AX,cPARP1,Area,Eccentricity,Roundness,CD11c.MY,CD15.MY,CD163.MP,CD207.MY,CD31.stromal,CD4.T.cells,CD68.MP,CD8.T.cells,Cancer,Other,Other.MY,Stromal,T.regs,B.cells,Other.immune'
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

def create_heatmap_of_f1(all_results,cancer1_results,cancer2_results,non_cancer_results, therapy):

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

    # Define the custom sorting order for 'Class_Pair'
    class_pair_order = ['BRCAloss_vs_HRP', 'BRCAloss_vs_CCNE1amp', 'HRP_vs_HRD', 'BRCAloss_vs_HRD', 'CCNE1amp_vs_HRD', 'HRP_vs_CCNE1amp']

    # Convert 'Class_Pair' to Categorical with custom sorting order
    combined_data['ClassPair'] = pd.Categorical(combined_data['ClassPair'], categories=class_pair_order, ordered=True)

    # Reorganize the DataFrame
    combined_data = combined_data.sort_values(by='ClassPair')
    
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
    custom_yticks = ['BRCAloss or HRP', 'BRCAloss or CCNE1amp', 'HRD or HRP', 'BRCAloss or HRD', 'HRD or CCNE1amp', 'HRP or CCNE1amp',]  # Example: replace with your custom y ticks
    heatmap.set_yticklabels(custom_yticks, rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tick_params(left = False, bottom = False)
        
    plt.title(therapy)
    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/preds_heatmap_{therapy}.svg")
    plt.show()
    
def prep_df(df):
    
    # Choose features to work with 
    features = ['TIM3','pSTAT1','CD45RO','CD20','CD11c','CD207','GranzymeB','CD163','CD4','CD3d','CD8a','FOXP3','PD1','CD15','PDL1_488','Ki67','Vimentin','MHCII','MHCI','ECadherin','aSMA','CD31','pTBK1','CK7','yH2AX','cPARP1', 'Area','Eccentricity','Roundness','GlobalCellType','Molecular.profile2', 'therapy_sequence','cycif.slide']
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

def create_roc_plot(roc_data_list, title): 

    # Check for consistent lengths
    lengths_fpr = {len(df['FPR']) for df in roc_data_list}
    lengths_tpr = {len(df['TPR']) for df in roc_data_list}

    # Choose the maximum length as common
    common_length = max(max(lengths_fpr), max(lengths_tpr))

    # Interpolate FPR and TPR to a common length
    interp_fpr = np.linspace(0, 1, common_length)
    interp_tpr_list = [interp1d(df['FPR'], df['TPR'], kind='linear', fill_value='extrapolate')(interp_fpr) for df in roc_data_list]

    # Calculate Mean ROC Curve
    mean_fpr = interp_fpr
    mean_tpr = np.mean(interp_tpr_list, axis=0)
    mean_auc = np.mean([df['AUC'].values[0] for df in roc_data_list])

    # Calculate Confidence Interval
    confidence_interval = stats.t.interval(0.95, len(roc_data_list)-1,
                                        loc=mean_tpr, scale=stats.sem(interp_tpr_list, axis=0))

    # Plot the ROC Curve
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.fill_between(mean_fpr, confidence_interval[0], confidence_interval[1], color='b', alpha=0.2, label='95% CI')

    # Add labels and legend
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend()
    
    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles//mol_profile_{title}.svg")

    # Show the plot
    plt.show()
    
# Top 5 most predictive features
def plot_top_pred_features(df, column, xlabel, title, return_values = False, return_plot = False):
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
    top_features = sorted_features[:5]

    # Ensure all lists have the same length by padding with np.nan
    max_length = max(len(feature_importance_dict[feature]) for feature in top_features)
    for feature in top_features:
        feature_importance_dict[feature] += [np.nan] * (max_length - len(feature_importance_dict[feature]))

    # Create a DataFrame containing only the top 5 features and their values
    data = pd.DataFrame({feature: feature_importance_dict[feature] for feature in top_features})
    
    if return_plot == True:
        
        if title == 'Cancer_BRCAloss_vs_HRP_PDS':
        
            # Create a single plot with all top features
            plt.figure(figsize=(3, 3))

            # Create a boxplot for all top features in a single plot
            sns.boxplot(data=data,width=.5, orient = 'h', color = '#fed18c') #  #d2d2e6
            
            # Check significance and annotate the plot
            for i in range(len(top_features) - 1):
                feature1 = top_features[i]
                feature2 = top_features[i+1]
                p_value = mannwhitneyu(data[feature1].dropna(), data[feature2].dropna()).pvalue
                if p_value < 0.05:
                    plt.text(0.09, i + 0.3, "*", ha='center', va='center', color='black', fontsize=30)
                    
            # Add labels and title
            plt.xlabel(xlabel)
            plt.xlim(-0.01, 0.1)
            plt.ylabel('Top Features')
            plt.title(title)

            # Adjust layout
            plt.tight_layout()

            plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/mol_profile_{title}.svg")
            
            # Show the plot
            plt.show()

    return data.median().to_dict() if return_values == True else top_features

def create_spider_plot(cells_df, class1, class2, cell_type, therapy, top_markers):
    
    if 'BRCAloss' in class1:
        index = class1.index('BRCAloss')
        class1[index] = 'BRCAmutmet'

    df_class1 = cells_df[cells_df['Molecular.profile2'].isin(class1)]
    df_class2 = cells_df[cells_df['Molecular.profile2'].isin(class2)]

    if cell_type == "Non-cancer":
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others', 'Cancer'])]

    elif cell_type in ["Cancer", "Stromal"]:
        celltype_df1 = df_class1[df_class1["GlobalCellType"] == cell_type]
        celltype_df2 = df_class2[df_class2["GlobalCellType"] == cell_type]
        
    elif cell_type == "Immune":
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        
    elif cell_type == 'All':
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others'])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others'])]

    final_df1 = celltype_df1[celltype_df1["therapy_sequence"] == therapy]
    final_df2 = celltype_df2[celltype_df2["therapy_sequence"] == therapy]
    
    markers_dict = {'Mol profiles': [f'{class1} vs {class2}']}
    
    for marker in top_markers:
        mean1 = final_df1[marker].mean()
        mean2 = final_df2[marker].mean()

        # Print statements for debugging
        # print(f"Marker: {marker}, Mean1: {mean1}, Mean2: {mean2}")

        # Check if mean2 is zero or close to zero to avoid division by zero
        ratio = mean1 / mean2 if mean2 != 0 else 1

        markers_dict[marker] = [ratio]

    mean_df = pd.DataFrame(markers_dict)
    
    df = pd.DataFrame(dict(r=mean_df.iloc[0, 1:].values.tolist(),
                        theta=top_markers))
    
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')

    # Update the layout to increase text size
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {'size': 32},  # Set the font size to your preferred value
        'polar': {'radialaxis': {'range': [0, 2.2], 'tickvals': [0.5, 1.0, 1.5]}}
    })

    class1_name = "".join(class1)
    class2_name = "".join(class2)
    
    fig.write_image(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/{cell_type}_{class1_name}_{class2_name}_{therapy}_{top_markers[0]}.svg")
    fig.show()

def create_spider_plot_four_molprofiles(cells_df, class1, class2, class3, class4, cell_type, therapy, top_markers):
    
    if 'BRCAloss' in class1:
        index = class1.index('BRCAloss')
        class1[index] = 'BRCAmutmet'
        
    class1_name = "".join(class1)
    class2_name = "".join(class2)
    class3_name = "".join(class3)
    class4_name = "".join(class4)

    # Subset cells dataframe based on therapy, molecular profile and cell type
    final_cells_df = cells_df[cells_df["therapy_sequence"] == therapy]

    df_class1 = final_cells_df[final_cells_df['Molecular.profile2'].isin(class1)]
    df_class2 = final_cells_df[final_cells_df['Molecular.profile2'].isin(class2)]
    df_class3 = final_cells_df[final_cells_df['Molecular.profile2'].isin(class3)]
    df_class4 = final_cells_df[final_cells_df['Molecular.profile2'].isin(class4)]
    
    if cell_type == "Non-cancer":
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df3 = df_class3[~df_class3["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df4 = df_class4[~df_class4["GlobalCellType"].isin(['Others', 'Cancer'])]
        
    elif cell_type in ["Cancer", "Stromal"]:
        celltype_df1 = df_class1[df_class1["GlobalCellType"] == cell_type]
        celltype_df2 = df_class2[df_class2["GlobalCellType"] == cell_type]
        celltype_df3 = df_class3[df_class3["GlobalCellType"] == cell_type]
        celltype_df4 = df_class4[df_class4["GlobalCellType"] == cell_type]
        
    elif cell_type == "Immune":
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df3 = df_class3[~df_class3["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df4 = df_class4[~df_class4["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        
    elif cell_type == 'All':
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others'])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others'])]
        celltype_df3 = df_class3[~df_class3["GlobalCellType"].isin(['Others'])]
        celltype_df4 = df_class4[~df_class4["GlobalCellType"].isin(['Others'])]
    
    # Create a dictionary to collect median values for molecular profiles
    markers_dict = {'Molecular profiles': [f'{class1_name}', f'{class2_name}', f'{class3_name}', f'{class4_name}']}
    
    for marker in top_markers:
        median1 = np.median(celltype_df1[marker].values)
        median2 = np.median(celltype_df2[marker].values)
        median3 = np.median(celltype_df3[marker].values)
        median4 = np.median(celltype_df4[marker].values)

        markers_dict[marker] = [median1, median2, median3, median4]

    # Prepare dictionary as dataframe for plotting
    mean_df = pd.DataFrame(markers_dict)
    melted_df = pd.melt(mean_df, id_vars=['Molecular profiles'], var_name='theta', value_name='r')
    
    # Plot median expression across molecular profiles
    fig = px.line_polar(melted_df, r='r', theta='theta', color = 'Molecular profiles', 
                        line_close=True, color_discrete_sequence = ['#D33F49','#FED18C','#508791','#A7AEF2'])

    # Update the layout to increase text size
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {'size': 20},  # Set the font size to your preferred value
        'polar': {'radialaxis': {'range': [min(melted_df['r']), max(melted_df['r'])]}} # , 'tickvals': [0, 1, 2]
    })
    
    fig.write_image(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/{cell_type}_{class1_name}_vs_{class2_name}_{class3_name}_{class4_name}_{therapy}_{top_markers[0]}.svg")
    fig.show()

def create_violin_plot(cells_df, class1, class2, cell_type, therapy, top_markers):
    if 'BRCAloss' in class1:
        index = class1.index('BRCAloss')
        class1[index] = 'BRCAmutmet'

    # Filter data based on classes
    df_class1 = cells_df[cells_df['Molecular.profile2'].isin(class1)]
    df_class2 = cells_df[cells_df['Molecular.profile2'].isin(class2)]

    # Filter data based on cell type
    if cell_type == "Non-cancer":
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others', 'Cancer'])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others', 'Cancer'])]
    elif cell_type in ["Cancer", "Stromal"]:
        celltype_df1 = df_class1[df_class1["GlobalCellType"] == cell_type]
        celltype_df2 = df_class2[df_class2["GlobalCellType"] == cell_type]
    elif cell_type == "Immune":
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
    elif cell_type == 'All':
        celltype_df1 = df_class1[~df_class1["GlobalCellType"].isin(['Others'])]
        celltype_df2 = df_class2[~df_class2["GlobalCellType"].isin(['Others'])]

    # Filter data based on therapy
    final_df1 = celltype_df1[celltype_df1["therapy_sequence"] == therapy]
    final_df2 = celltype_df2[celltype_df2["therapy_sequence"] == therapy]

    class1_name = "".join(class1)
    class2_name = "".join(class2)

    # Concatenate data for all markers
    all_markers_data = pd.concat([final_df1[top_markers].stack(), final_df2[top_markers].stack()], axis=1)
    all_markers_data.columns = [class1_name, class2_name]
    all_markers_data['Marker'] = all_markers_data.index.get_level_values(1)

    # Reset index to ensure 'Marker' column is a regular column
    all_markers_data.reset_index(drop=True, inplace=True)

    # Melt the dataframe to long format
    all_markers_data = all_markers_data.melt(id_vars=['Marker'], value_vars=[class1_name, class2_name], var_name='Molecular profiles', value_name='Pre-processed expression')

    # Define the figure arguments and test configuration
    fig_args = {'x': 'Marker',
            'y': 'Pre-processed expression',
            'hue':'Molecular profiles',
            'data': all_markers_data,
            'order': top_markers,
            'hue_order':[class1_name,class2_name]}

    configuration = {'test':'Mann-Whitney',
                    'comparisons_correction':None,
                    'text_format':'star'}

    ax = sns.violinplot(**fig_args, palette = {class1_name: "#fed18c", class2_name: "#d2d2e6"}, fill=False)

    significanceComparisons = [
        ((marker, class1_name), (marker, class2_name))
        for marker in top_markers
    ]
    annotator = Annotator(ax, significanceComparisons, **fig_args, plot = 'violinplot')
    annotator.configure(**configuration).apply_test().annotate()

    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/{cell_type}_{class1_name}_{class2_name}_{therapy}_{top_markers[0]}.svg")
    plt.show()

def create_heatmap_mean_value_per_marker(cells_df, therapy, cell_marker_df):
    
    # Drop duplicates in marker dataframe
    cell_marker_df = cell_marker_df.drop_duplicates()

    # Pre-process cells dataframe
    cells_df = prep_df(cells_df)
    # Subset cells dataframe based on therapy and molecular profile
    df = cells_df[cells_df["therapy_sequence"] == therapy]
    ccne1amp_df = df[df['Molecular.profile2'] == 'CCNE1amp']
    hrp_df = df[df['Molecular.profile2'] == 'HRP']
    brcamutmet_df = df[df['Molecular.profile2']=='BRCAmutmet']
    hrd_df = df[df['Molecular.profile2']=='HRD']

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame(columns=['CellType', 'Marker', 'CCNE1amp', 'HRP', 'BRCAloss', 'HRD', 'PvalueFromKruskal'])

    for index, row in cell_marker_df.iterrows():

        cell_type = row['CellType']
        marker = row['Marker']
        
        # Subset based on cell type
        if cell_type == 'Immune-stromal':
            celltype_ccne1amp_df = ccne1amp_df[~ccne1amp_df["GlobalCellType"].isin(['Others', 'Cancer'])]
            celltype_hrp_df = hrp_df[~hrp_df["GlobalCellType"].isin(['Others', 'Cancer'])]
            celltype_brcamutmet_df = brcamutmet_df[~brcamutmet_df["GlobalCellType"].isin(['Others', 'Cancer'])]
            celltype_hrd_df = hrd_df[~hrd_df["GlobalCellType"].isin(['Others', 'Cancer'])]
        elif cell_type in ["Cancer", "Stromal"]:
            celltype_ccne1amp_df = ccne1amp_df[ccne1amp_df["GlobalCellType"] == cell_type]
            celltype_hrp_df = hrp_df[hrp_df["GlobalCellType"] == cell_type]
            celltype_brcamutmet_df = brcamutmet_df[brcamutmet_df["GlobalCellType"] == cell_type]
            celltype_hrd_df = hrd_df[hrd_df["GlobalCellType"] == cell_type]
        elif cell_type == "Immune":
            celltype_ccne1amp_df = ccne1amp_df[~ccne1amp_df["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
            celltype_hrp_df = hrp_df[~hrp_df["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
            celltype_brcamutmet_df = brcamutmet_df[~brcamutmet_df["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
            celltype_hrd_df = hrd_df[~hrd_df["GlobalCellType"].isin(['Others', 'Cancer', "Stromal"])]
        elif cell_type == 'All':
            celltype_ccne1amp_df = ccne1amp_df[~ccne1amp_df["GlobalCellType"].isin(['Others'])]
            celltype_hrp_df = hrp_df[~hrp_df["GlobalCellType"].isin(['Others'])]
            celltype_brcamutmet_df = brcamutmet_df[~brcamutmet_df["GlobalCellType"].isin(['Others'])]
            celltype_hrd_df = hrd_df[~hrd_df["GlobalCellType"].isin(['Others'])]

        # Extract marker values for each group
        values_ccne1amp_df = celltype_ccne1amp_df[marker].values
        values_hrp_df = celltype_hrp_df[marker].values
        values_brcamutmet_df = celltype_brcamutmet_df[marker].values
        values_hrd_df = celltype_hrd_df[marker].values

        # Perform Kruskal-Wallis H-test
        h_stat, p_value = stats.kruskal(values_ccne1amp_df, values_hrp_df, values_brcamutmet_df, values_hrd_df)

        # Add the results to the result_df
        result_df = result_df.append({'CellType': cell_type, 'Marker': marker,
                                    'CCNE1amp': np.median(values_ccne1amp_df), 'HRP': np.median(values_hrp_df),
                                    'BRCAloss': np.median(values_brcamutmet_df),'HRD': np.median(values_hrd_df),
                                    'PvalueFromKruskal': p_value},
                                    ignore_index=True)

    # Add a column indicating whether p-value is less than 0.05
    result_df['Significance'] = result_df['PvalueFromKruskal'] < 0.05

    # Combine CellType and Marker columns into a new column
    result_df['CellMarker'] = result_df['CellType'] + '_' + result_df['Marker']

    # Set the new column 'CellMarker' as the index
    result_df.set_index('CellMarker', inplace=True)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    for cell_marker, row in result_df.iterrows():
        # Extract the values to scale
        values_to_scale = row[['BRCAloss', 'HRD', 'HRP', 'CCNE1amp']].values.reshape(-1, 1)

        # Fit and transform the values
        scaled_values = scaler.fit_transform(values_to_scale)

        # Update the specified columns in result_df with the scaled values
        result_df.loc[cell_marker, ['BRCAloss', 'HRD', 'HRP', 'CCNE1amp']] = scaled_values.flatten()

    # Create a heatmap
    plt.figure(figsize=(3, 5))
    ax = sns.heatmap(result_df[['BRCAloss', 'HRD', 'HRP', 'CCNE1amp']], 
                    vmin=result_df[['BRCAloss', 'HRD', 'HRP', 'CCNE1amp']].values.min(), vmax=result_df[['BRCAloss', 'HRD', 'HRP', 'CCNE1amp']].values.max(),
                    annot=False, cmap=LinearSegmentedColormap.from_list('custom_cmap', ['#6599CD','#FFFFFF','#D33F49']), # '#fcedee',
                    cbar_kws={'label': 'Median expression values'})

    # Add white lines to visually separate the two groups of columns
    for i in range(1, 3):
        ax.axvline(i * 2, color='white', linewidth=2)

    # Add * for significance
    for i, (cell_marker, row) in enumerate(result_df.iterrows()):
        if row['Significance']:
            plt.text(4, i + 0.6, "*", ha='center', va='center', color='black', fontsize=30)

    ax.set(xlabel="", ylabel="Cell type & Marker")
    plt.title(therapy)
    plt.xticks(rotation=45, ha='right')  # Rotate xticks and align them to the right
    plt.tick_params(left = False, bottom = False) 

    # plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/mol_profile_predictive_features_{therapy}.svg") 

    plt.show()
    
def plot_oncogrid_of_pred_features(result_df, result_type, therapy):
    
    result_df = result_df.transpose()

    # Calculate the count of non-null values in each row
    non_null_counts = result_df.notnull().sum(axis=1)

    # Sort the DataFrame based on the count of non-null values in each row
    df_sorted = result_df.iloc[non_null_counts.argsort()[::-1]]

    # if result_type == 'clinical':
    #     df_sorted = df_sorted.drop(['refage', 'figo_stage'])

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
    if result_type == 'mol_profile': 
        xtick_labels = ['BRCAloss or HRP','BRCAloss or CCNE1amp','HRD or HRP',
                        'BRCAloss or HRP','BRCAloss or CCNE1amp','HRD or HRP',
                        'BRCAloss or HRP','BRCAloss or CCNE1amp','HRD or HRP',
                        'BRCAloss or HRP','BRCAloss or CCNE1amp','HRD or HRP']
    elif result_type == 'clinical':
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

    if result_type == 'mol_profile':
        for i in range(1, 9):
            heatmap.axvline(i * 3, color='white', linewidth=5)
    elif result_type == 'clinical':
        for i in range(1, 6):
            heatmap.axvline(i * 2, color='white', linewidth=5)


    heatmap.set_xticklabels(xtick_labels, rotation=45, ha='right')  # Set custom xtick labels
    plt.tick_params(left = False, bottom = False) 

    plt.savefig(f"/Users/alex/Desktop/Laboratory/Projects/Thesis/Results/Molprofiles/{result_type}_oncogrid_of_predictive_features_{therapy}.svg")
    plt.show()

def process_results(cells_df, chosen_results, folder_path, result_type, result_df, final_plot = False):
    
    results_dict = {}
    
    # Collect predictive features for corresponding comparison
    for chosen_comparison in chosen_results:

        cell_type, class1, _, class2, therapy = chosen_comparison.split("_")

        os.chdir(folder_path)

        if result_type in ['Cancer','Cancer1','Cancer2']:
            df = pd.read_csv(f'{chosen_comparison}.csv')
        else:
            df = pd.concat([pd.read_csv(f'{chosen_comparison}_1.csv'), pd.read_csv(f'{chosen_comparison}_2.csv')], axis=0)

        if final_plot == 'oncogrid':
            median_values = plot_top_pred_features(df, 'permutation_scores', 'MDA', chosen_comparison, return_values= True)
            results_dict[f'{result_type}_{class1}_vs_{class2}'] = median_values

        elif final_plot == 'heatmap':
            top_5_features = plot_top_pred_features(df, 'permutation_scores', 'MDA', chosen_comparison, return_plot=True)
            results_dict[chosen_comparison] = top_5_features
            
            if chosen_comparison == 'Cancer_BRCAloss_vs_HRP_PDS':
                
                # Create ROC plot
                roc_files = [
                    filename for filename in os.listdir(folder_path)
                    if filename.startswith(chosen_comparison) and filename.endswith('roc_curve_data.csv')
                ]

                roc_data_list = [pd.read_csv(filepath) for filepath in roc_files]
                create_roc_plot(roc_data_list, chosen_comparison)

                # Create spider plot 
                create_spider_plot(cells_df, [class1], [class2], result_type, therapy, top_5_features)

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