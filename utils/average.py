import pandas as pd

name_normal = 'results_dependent_feature_selection_mushroom_True_whole_dataset.csv'
name_acc = 'results_dependent_feature_selection_f1_acc_mushroom_True_whole_dataset.csv'

df = pd.read_csv(name_normal,
                 header=0,
                 sep='\t',
                 names=["index", "highest_proba", "outcome", "cost_of_classification",
                        "used_features"])

df_acc = pd.read_csv(name_acc,
                     sep='\t',
                     names=["accuracy", "F1"],
                     header=0)

averag = pd.DataFrame(df["cost_of_classification"].mean(), columns=["average_cost_of_classification"], index=[0])

df_acc = pd.concat([df_acc, averag], axis=1)

df_acc.to_csv(name_acc, sep='\t', encoding='utf-8', mode='w', header=True, index=False)
