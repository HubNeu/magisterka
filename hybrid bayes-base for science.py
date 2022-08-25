#!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold
import numpy as np

np.set_printoptions(suppress=True)
import pandas as pd
import scipy
import random
import copy
import warnings
from multiprocessing import Process
import os

warnings.filterwarnings("ignore")

# print("Libs imported. Python version is: ", python_version())

# utility functions

cols_mushroom = [
    "labels",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "population",
    "habitat",
]

cols_car = ["buying", "maintenance", "doors", "passengers", "boot", "safety", "labels"]

cols_audiology = [
    "age_gt_60",
    "air",
    "airBoneGap",
    "ar_c",
    "ar_u",
    "bone",
    "boneAbnormal",
    "bser",
    "history_buzzing",
    "history_dizziness",
    "history_fluctuating",
    "history_fullness",
    "history_heredity",
    "history_nausea",
    "history_noise",
    "history_recruitment",
    "history_ringing",
    "history_roaring",
    "history_vomiting",
    "late_wave_poor",
    "m_at_2k",
    "m_cond_lt_1k",
    "m_gt_1k",
    "m_m_gt_2k",
    "m_m_sn",
    "m_m_sn_gt_1k",
    "m_m_sn_gt_2k",
    "m_m_sn_gt_500",
    "m_p_sn_gt_2k",
    "m_s_gt_500",
    "m_s_sn",
    "m_s_sn_gt_1k",
    "m_s_sn_gt_2k",
    "m_s_sn_gt_3k",
    "m_s_sn_gt_4k",
    "m_sn_2_3k",
    "m_sn_gt_1k",
    "m_sn_gt_2k",
    "m_sn_gt_3k",
    "m_sn_gt_4k",
    "m_sn_gt_500",
    "m_sn_gt_6k",
    "m_sn_lt_1k",
    "m_sn_lt_2k",
    "m_sn_lt_3k",
    "middle_wave_poor",
    "mod_gt_4k",
    "mod_mixed",
    "vmod_s_mixed",
    "mod_s_sn_gt_500",
    "mod_sn",
    "mod_sn_gt_1k",
    "mod_sn_gt_2k",
    "mod_sn_gt_3k",
    "mod_sn_gt_4k",
    "mod_sn_gt_500",
    "notch_4k",
    "notch_at_4k",
    "o_ar_c",
    "o_ar_u",
    "s_sn_gt_1k",
    "s_sn_gt_2k",
    "s_sn_gt_4k",
    "speech",
    "static_normal",
    "tymp",
    "viith_nerve_signs",
    "wave_V_delayed",
    "waveform_ItoV_prolonged",
    "p_index",
    "labels",
]

"""
print("Cols audiology:",audiology_cost.isnull().values.any())
print("Cols car", car_cost.isnull().values.any())
print("cols_mushroom:", mushroom_cost.isnull().values.any())
print("")
"""

"""
https://archive.ics.uci.edu/ml/datasets/car+evaluation
0-5 -> data
6 -> labels
"""


def load_car():
    df_car = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        header=None,
        names=cols_car,
    )
    # mappings using indexes:
    # X = df_car.loc[:, :5].values
    # y = df_car.loc[:, 6].values
    df_car = df_car.drop("buying", axis=1)
    df_car = df_car.drop("maintenance", axis=1)
    labels_col = df_car.pop("labels")
    df_car.insert(0, "labels", labels_col)
    return df_car


"""
https://archive.ics.uci.edu/ml/datasets/mushroom
1-22 -> data
0 -> labels
"""


def load_mushroom():
    df_mushroom = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        header=None,
        names=cols_mushroom,
    )
    # index mappings
    # X = df_mushroom.loc[:, 1:].values
    # y = df_mushroom.loc[:, 0].values
    # drop values corelating a bit too much like this
    df_mushroom = df_mushroom.drop("odor", axis=1)
    df_mushroom = df_mushroom.drop("spore_print_color", axis=1)
    return df_mushroom


"""
https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29
0:length-2 -> data
length-1 unique id (p1-p200)
length -> labels
"""


def load_audiology():
    df_audiology = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.data",
        header=None,
        names=cols_audiology,
    )
    # index mapping
    # length = len(df_audiology.columns)
    # X = df_audiology.loc[:, : length - 3].values
    # y = df_audiology.loc[:, length - 1].values
    df_audiology = df_audiology.drop("p_index", axis=1)
    # cols to drop and try for modified datasets:
    df_audiology = df_audiology.drop("age_gt_60", axis=1)
    df_audiology = df_audiology.drop("speech", axis=1)
    labels_col = df_audiology.pop("labels")
    df_audiology.insert(0, "labels", labels_col)
    return df_audiology


# Choose dataset
# dataset = load_car()
# dataset_name = "car"
# dataset_costs = car_cost

# dataset = load_mushroom()
# dataset_name = "mushroom"
# dataset_costs = mushroom_cost

dataset = load_audiology()
dataset_name = "audiology"
# dataset_costs = audiology_cost
# print("dataset.info()")
# print(dataset.info())
# print(">>dataset.describe()")
# print(dataset.describe())
# for c in dataset.columns.tolist():
    # print(">>dataset[", c, "].value_counts()")
    # print(dataset[c].value_counts())
# print("First five records:")
# print(dataset.head())

# Extract to X and y
X_cat = dataset.loc[:, dataset.columns != "labels"]
y_cat = dataset.loc[:, "labels"]

# print("Size of X: ", np.shape(X_cat))
# print("Size of y: ", np.shape(y_cat))

# print("Size of dataset costs: ", np.shape(dataset_costs))
# print("Cost of classification on full dataset: ", dataset_costs.sum(axis=1)[0])
# print("Labels encoded: ", np.shape(encoded_y_train), ", ", np.shape(encoded_y_test))

# Collectors of values

cols_one_hot = [
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "habitat",
    "age_gt_60",
    "airBoneGap",
    "boneAbnormal",
    "history_buzzing",
    "history_dizziness",
    "history_fluctuating",
    "history_fullness",
    "history_heredity",
    "history_nausea",
    "history_noise",
    "history_recruitment",
    "history_ringing",
    "history_roaring",
    "history_vomiting",
    "late_wave_poor",
    "m_at_2k",
    "m_cond_lt_1k",
    "m_gt_1k",
    "m_m_gt_2k",
    "m_m_sn",
    "m_m_sn_gt_1k",
    "m_m_sn_gt_2k",
    "m_m_sn_gt_500",
    "m_p_sn_gt_2k",
    "m_s_gt_500",
    "m_s_sn",
    "m_s_sn_gt_1k",
    "m_s_sn_gt_2k",
    "m_s_sn_gt_3k",
    "m_s_sn_gt_4k",
    "m_sn_2_3k",
    "m_sn_gt_1k",
    "m_sn_gt_2k",
    "m_sn_gt_3k",
    "m_sn_gt_4k",
    "m_sn_gt_500",
    "m_sn_gt_6k",
    "m_sn_lt_1k",
    "m_sn_lt_2k",
    "m_sn_lt_3k",
    "middle_wave_poor",
    "mod_gt_4k",
    "mod_mixed",
    "vmod_s_mixed",
    "mod_s_sn_gt_500",
    "mod_sn",
    "mod_sn_gt_1k",
    "mod_sn_gt_2k",
    "mod_sn_gt_3k",
    "mod_sn_gt_4k",
    "mod_sn_gt_500",
    "notch_4k",
    "notch_at_4k",
    "s_sn_gt_1k",
    "s_sn_gt_2k",
    "s_sn_gt_4k",
    "static_normal",
    "viith_nerve_signs",
    "wave_V_delayed",
    "waveform_ItoV_prolonged",
]

cols_ordinal = [
    "buying",
    "maintenance",
    "doors",
    "passengers",
    "boot",
    "safety",
    "population",
    "air",
    "ar_c",
    "ar_u",
    "bser",
    "bone",
    "o_ar_c",
    "o_ar_u",
    "speech",
    "tymp",
]

# print("Cols created.")

# Make order of categories per each column in ordinal_columns
order_of_ordinal_categories = pd.DataFrame.from_dict(
    {
        "buying": ["low", "med", "high", "vhigh", "filler1", "filler2", "filler3"],
        "maintenance": ["low", "med", "high", "vhigh", "filler1", "filler2", "filler3"],
        "doors": ["2", "3", "4", "5more", "filler1", "filler2", "filler3"],
        "passengers": ["2", "4", "more", "filler1", "filler2", "filler3", "filler4"],
        "boot": ["small", "med", "big", "filler1", "filler2", "filler3", "filler4"],
        "safety": ["low", "med", "high", "filler1", "filler2", "filler3", "filler4"],
        "population": ["y", "v", "s", "n", "c", "a", "filler1"],
        "air": [
            "normal",
            "mild",
            "moderate",
            "severe",
            "profound",
            "filler1",
            "filler2",
        ],
        "ar_c": ["?", "absent", "normal", "elevated", "filler1", "filler2", "filler3"],
        "ar_u": ["?", "absent", "normal", "elevated", "filler1", "filler2", "filler3"],
        "bser": ["?", "normal", "degraded", "filler1", "filler2", "filler3", "filler4"],
        "bone": ["?", "unmeasured", "normal", "mild", "moderate", "filler1", "filler3"],
        "o_ar_c": [
            "?",
            "absent",
            "normal",
            "elevated",
            "filler1",
            "filler2",
            "filler3",
        ],
        "o_ar_u": [
            "?",
            "absent",
            "normal",
            "elevated",
            "filler1",
            "filler2",
            "filler3",
        ],
        "speech": [
            "?",
            "unmeasured",
            "very_poor",
            "poor",
            "normal",
            "good",
            "very_good",
        ],
        "tymp": ["a", "as", "b", "ad", "c", "filler1", "filler2"],
    }
)


# print("Order created.")
# print(order_of_ordinal_categories)


# Create custom encoding categorical bayes classifier
class EncodingCategoricalBayes:
    def __init__(
            self,
            # classifier,
            ordinal_categories_order,
            ordinal_columns,
            one_hot_columns,
            dataset,
    ):
        # self.classifier = classifier
        self.ordinal_categories_order = ordinal_categories_order
        self.ordinal_columns = ordinal_columns
        self.one_hot_columns = one_hot_columns
        self.transformer_dataset = dataset

    def fit(self, X, y):
        self.classifier = CategoricalNB(min_categories=X.shape[0])
        self.column_transformer = self.make_column_transformer(self.transformer_dataset)
        self.column_transformer.fit(self.transformer_dataset[X.columns.tolist()])
        return self.classifier.fit(self.encode_features(X), y)
        # return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(self.encode_features(X))
        # return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.encode_features(X))
        # return self.classifier.predict_proba(X)

    def encode_features(self, X):
        encoded_X = self.column_transformer.transform(X)
        if scipy.sparse.issparse(encoded_X):
            encoded_X = encoded_X.toarray()
        return encoded_X

    def make_column_transformer(self, X):
        # Get current ordinal and one hot columns
        total_column_list = X.select_dtypes(include="object").columns
        # print("Total col list: ", total_column_list)
        current_columns_one_hot = self.collect_current_one_hot_columns(
            total_column_list
        )
        current_columns_ordinal = self.collect_current_ordinal_columns(
            total_column_list
        )

        current_ordinal_col_ordering_to_encode = self.calculate_current_order_of_ordinal_columns_to_encode(
            current_columns_ordinal
        )

        # Create column transformer
        column_transformer = make_column_transformer(
            (OneHotEncoder(), current_columns_one_hot),
            (
                OrdinalEncoder(categories=current_ordinal_col_ordering_to_encode),
                current_columns_ordinal,
            ),
        )
        return column_transformer

    def calculate_current_order_of_ordinal_columns_to_encode(self, argColumns):
        # Get common cols to feed them in proper order to ordinal encoder
        index_of_common_cols = self.ordinal_categories_order.columns.intersection(
            argColumns
        )
        # Convert to list
        order_of_ordinal_categories_list = (
            self.ordinal_categories_order[index_of_common_cols]
            .values.transpose()
            .tolist()
        )
        return order_of_ordinal_categories_list

    def intersection(self, lst1, lst2):
        # collects common elements in both lists
        return [value for value in lst1 if value in lst2]

    def collect_current_one_hot_columns(self, argCols):
        return self.intersection(self.one_hot_columns, argCols)

    def collect_current_ordinal_columns(self, argCols):
        # make list of all values and create steps for them
        return self.intersection(self.ordinal_columns, argCols)

    def get_params(self, deep=True):
        return self.classifier.get_params()


# print("Class EncodingCategoricalBayes has been created")

# Create a Bayes Classifier || requires min_categories due to a bug with indexes, reporting the bug added to TODO
enb = EncodingCategoricalBayes(
    ordinal_categories_order=order_of_ordinal_categories,
    ordinal_columns=cols_ordinal,
    one_hot_columns=cols_one_hot,
    dataset=X_cat,
)

max_seed_val = 2 ** 32 - 1
random_seed_kfold = random.randrange(0, max_seed_val)
skf = StratifiedKFold(n_splits=10, random_state=random_seed_kfold, shuffle=True)
le = LabelEncoder().fit(y_cat)
reference_outcomes = le.transform(y_cat)
i = 0
all_features_list = X_cat.columns.tolist()

final_result_dataframe = pd.DataFrame({'highest_proba': pd.Series(dtype='float'),
                                       'outcome': pd.Series(dtype='int'),
                                       'cost_of_classification': pd.Series(dtype='int'),
                                       'used_features': pd.Series(dtype='str')})

for train_idx, test_idx in skf.split(X_cat, y_cat):
    X_train, X_test = (
        X_cat.iloc[train_idx],
        X_cat.iloc[test_idx],
    )

    # y
    y_train, y_test = (
        y_cat[train_idx],
        y_cat[test_idx],
    )
    # print("Data has been split.")
    # print("X contains features: ", X_train.columns == "index")

    # Transform y using label encoder
    encoded_y_train = le.transform(y_train)
    encoded_y_test = le.transform(y_test)

    # Train the model using the training sets
    enb.fit(X_train, encoded_y_train)

    # Predict the response for test dataset
    y_pred = enb.predict(X_test)
    y_pred_probas = enb.predict_proba(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of ", i, " fold:", metrics.accuracy_score(encoded_y_test, y_pred) * 100, "%")
    print("F1 score", "of ", i, " fold:", metrics.f1_score(encoded_y_test, y_pred, average="weighted") * 100, "%")
    i += 1

    outcomes = pd.DataFrame(y_pred, columns=["outcome"], index=test_idx)

    highest_probas = pd.DataFrame(np.max(np.max(y_pred_probas, axis=1), axis=0), columns=["highest_proba"],
                                  index=test_idx)

    cost_of_classification = pd.DataFrame(
        len(X_train.columns.tolist()),
        columns=["cost_of_classification"],
        index=test_idx
    )

    iter_features = pd.DataFrame(",".join(all_features_list),
                                 columns=["used_features"],
                                 index=test_idx)

    batch_result_dataframe = pd.concat(
        [
            outcomes,
            highest_probas,
            cost_of_classification,
            iter_features
        ],
        axis=1
    )
    final_result_dataframe = pd.concat(
        [final_result_dataframe, batch_result_dataframe]
    )

final_result_dataframe.sort_index(inplace=True)
all_outcomes = final_result_dataframe["outcome"].astype(dtype='int32')
print("Total accuracy:", metrics.accuracy_score(reference_outcomes, all_outcomes) * 100, "%")
print("Total F1 score:", metrics.f1_score(reference_outcomes, all_outcomes, average="weighted") * 100, "%")
combined_results_acc = pd.DataFrame([(
    metrics.accuracy_score(reference_outcomes, all_outcomes),
    metrics.f1_score(reference_outcomes, all_outcomes, average="weighted"),
    final_result_dataframe["cost_of_classification"].mean())
],
    columns=["accuracy", "F1", "average_cost_of_classification"], index=[0])
final_result_dataframe.to_csv('results_base_' + dataset_name + '_.csv',
                              sep='\t', encoding='utf-8', mode='a',
                              header=True, index=False)
combined_results_acc.to_csv(
    'results_base_f1_acc_' + dataset_name + '_.csv', sep='\t',
    encoding='utf-8', mode='a',
    header=True, index=False)
