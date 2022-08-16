#!/usr/bin/env python
# coding: utf-8

"""
Solved:
    -It's possible for train-test split to split data in such a way, that
   after encoding, X_train and X_test have different numbers of features.
   Split has to be rerun to fix it. First encode, then split again?
   BUT IT STILL HAS TO BE ENCODED AND SPLIT BEFORE STARTING CROSS VALIDATION
   AND SEQUENTIAL FEATURE SELECTION. MAYBE APPEND DURING SPLIT AND THEN SPLIT
   AGAIN?
    -Improve feature encoding to have proper ordering instead of random numbers
    which currently influence classification accuracy:
    https://datascience.stackexchange.com/questions/72343/encoding-with-ordinalencoder-how-to-give-levels-as-user-input

Fishy:
    -check and check for data leakage (def: https://scikit-learn.org/stable/glossary.html)

Pickup: 
    -move over feature encoding to pd.get_dummies() if it can be adapted
    -put the classifier in SFS
TODO:
    -modify SFS to check each predict_proba and stop if threshold is hit
    -add cost counting to SFS wrapper
    -???
    -tests and profit???
    -report a bug with indexes when predicting X_test using audiology and 
"""

## SKLEARN
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold

import numpy as np

np.set_printoptions(suppress=True)
import pandas as pd
from platform import python_version

import scipy

import random

print("Libs imported. Python version is: ", python_version())


# utility functions

cols_mushroom = [
    "labels",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
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
    "p-index",
    "labels",
]

cols_wine = [
    "labels",
    "alcohol",
    "mallic-acid",
    "alcalinity",
    "ash",
    "magnesium",
    "total-phenols",
    "flavonids",
    "nonflavonid-phenols",
    "proanthocyanins",
    "color-intensity",
    "hue",
    "od-of-diluted-wines",
    "proline",
]

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
    labels_col = df_car.pop("labels")
    df_car.insert(0, "labels", labels_col)
    # replace 5more in doors to 5
    # df_car.loc[df_car['doors'] == '5more', 'doors'] = '5'
    # df_car["doors"] = pd.to_numeric(df_car["doors"])
    # replace more in passengers to 5
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
    df_audiology = df_audiology.drop("p-index", axis=1)
    labels_col = df_audiology.pop("labels")
    df_audiology.insert(0, "labels", labels_col)
    return df_audiology


"""
https://www.alldatascience.com/classification/wine-dataset-analysis-with-python/
1:length -> data
0 -> labels
"""


def load_wine():
    df_wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        header=None,
        names=cols_wine,
    )
    # index mappings
    # length = len(df_wine.columns)
    # X = df_wine.loc[:, 1:].values
    # y = df_wine.loc[:, 0].values
    return df_wine

# Choose dataset
# dataset = load_car()
dataset = load_mushroom()
# dataset = load_audiology()
### dataset = load_wine() # all cols numerical, doesn't work

print(dataset.info())
# print("First five records:")
# print(dataset.head())

# Extract to X and y
X_cat = dataset.loc[:, dataset.columns != "labels"]
y_cat = dataset.loc[:, "labels"]

print("Size of X: ", np.shape(X_cat))
print("Size of y: ", np.shape(y_cat))

# Generate a matrix of costs
max_cost_allowed = 10000

dataset_costs = pd.DataFrame(
    np.random.randint(0, max_cost_allowed, size=(1, np.shape(X_cat)[1])),
    columns=X_cat.columns,
)

print("Size of dataset costs: ", np.shape(dataset_costs))
print("Cost of classification on full dataset: ", dataset_costs.sum(axis=1)[0])

max_seed_val = 2 ** 32 - 1

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=random.randrange(0, max_seed_val),
)
print("Data has been split.")
# print("X contains features: ", X_train.columns == "index")

# Transform y using label encoder
le = LabelEncoder().fit(y_cat)
encoded_y_train = le.transform(y_train)
encoded_y_test = le.transform(y_test)
print("Labels encoded: ", np.shape(encoded_y_train), ", ", np.shape(encoded_y_test))

# Collectors of values

cols_one_hot = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
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

print("Order created.")
print(order_of_ordinal_categories)

# Create custom encoding categorical bayes classifier
class EncodingCategoricalBayes:
    def __init__(
        self,
        classifier,
        ordinal_categories_order,
        ordinal_columns,
        one_hot_columns,
        dataset,
    ):
        self.classifier = classifier
        self.ordinal_categories_order = ordinal_categories_order
        self.ordinal_columns = ordinal_columns
        self.one_hot_columns = one_hot_columns
        self.column_transformer = self.make_column_transformer(dataset).fit(dataset)

    def fit(self, X, y):
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


print("Class EncodingCategoricalBayes has been created")

# Create a Bayes Classifier || requires min_categories due to a bug with indexes, reporting the bug added to TODO
nbayes = CategoricalNB(min_categories=X_train.shape[0])

enb = EncodingCategoricalBayes(
    classifier=nbayes,
    ordinal_categories_order=order_of_ordinal_categories,
    ordinal_columns=cols_ordinal,
    one_hot_columns=cols_one_hot,
    dataset=X_cat,
)

# Train the model using the training sets
enb.fit(X_train, encoded_y_train)

# Predict the response for test dataset
y_pred = le.inverse_transform(enb.predict(X_test))
y_pred_probas = enb.predict_proba(X_test)
print(y_pred_probas[0:10])

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100, "%")
print("F1 score:", metrics.f1_score(y_test, y_pred, average="weighted") * 100, "%")

# Get cost of classification
classification_cost = dataset_costs[X_cat.columns].sum(axis=1)[0]
print("Classification cost per class:", classification_cost)
print("Classification cost of all classes:", classification_cost * np.shape(X_test)[0])

# save to csv
file_name = 'results dependent feature selection'
result.to_csv(file_name, sep='\t', encoding='utf-8')

