import pandas as pd
import numpy as np

# without
# datapath = "results\\sequential without data duplication\\modified datasets\\results_dependent_feature_selection_mushroom_False_modified_odor_spore_print_color.csv"
# with
datapath = "results\\sequential with data duplication\\modified datasets\\results_dependent_feature_selection_mushroom_True_modified_odor_spore_print_color.csv"

results = df = pd.read_csv(datapath,
                           header=0,
                           sep='\t',
                           names=["highest_proba", "outcome", "cost_of_classification",
                                  "used_features"])

print(">>np.unique(results['used_features'])")
print(np.unique(results['used_features']))
print()
print(">>results.groupby(['used_features'])['used_features'].count()")
print(results.groupby(["used_features"])["used_features"].count())
# print()
# print(">>np.column_stack(np.unique(results['used_features'].loc[results['used_features']=='doors'], return_counts=True))")
# print(np.column_stack(np.unique(results['highest_proba'].loc[results['used_features']=='doors'], return_counts=True)))
