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

TODO:
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
import scipy
import random
import copy
import warnings
import multiprocessing

warnings.filterwarnings("ignore")

# print("Libs imported. Python version is: ", python_version())

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
mushroom_cost = pd.DataFrame({
    "labels": 728,
    "cap-shape": 704,
    "cap-surface": 36,
    "cap-color": 624,
    "bruises": 717,
    "odor": 300,
    "gill-attachment": 38,
    "gill-spacing": 522,
    "gill-size": 4,
    "gill-color": 992,
    "stalk-shape": 999,
    "stalk-root": 14,
    "stalk-surface-above-ring": 838,
    "stalk-surface-below-ring": 726,
    "stalk-color-above-ring": 846,
    "stalk-color-below-ring": 190,
    "veil-type": 633,
    "veil-color": 176,
    "ring-number": 211,
    "ring-type": 186,
    "spore-print-color": 610,
    "population": 379,
    "habitat": 734
}, columns=cols_mushroom)

cols_car = ["buying", "maintenance", "doors", "passengers", "boot", "safety", "labels"]
car_cost = pd.DataFrame(
    {"buying": 250, "maintenance": 923, "doors": 200, "passengers": 733, "boot": 299, "safety": 808, "labels": 474
     }, columns=cols_car)

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

audiology_cost = pd.DataFrame({
    "age_gt_60": 119,
    "air": 399,
    "airBoneGap": 731,
    "ar_c": 323,
    "ar_u": 977,
    "bone": 796,
    "boneAbnormal": 107,
    "bser": 852,
    "history_buzzing": 326,
    "history_dizziness": 847,
    "history_fluctuating": 517,
    "history_fullness": 654,
    "history_heredity": 228,
    "history_nausea": 367,
    "history_noise": 973,
    "history_recruitment": 175,
    "history_ringing": 253,
    "history_roaring": 294,
    "history_vomiting": 851,
    "late_wave_poor": 901,
    "m_at_2k": 167,
    "m_cond_lt_1k": 840,
    "m_gt_1k": 97,
    "m_m_gt_2k": 352,
    "m_m_sn": 836,
    "m_m_sn_gt_1k": 201,
    "m_m_sn_gt_2k": 948,
    "m_m_sn_gt_500": 418,
    "m_p_sn_gt_2k": 137,
    "m_s_gt_500": 804,
    "m_s_sn": 173,
    "m_s_sn_gt_1k": 980,
    "m_s_sn_gt_2k": 871,
    "m_s_sn_gt_3k": 393,
    "m_s_sn_gt_4k": 446,
    "m_sn_2_3k": 292,
    "m_sn_gt_1k": 579,
    "m_sn_gt_2k": 987,
    "m_sn_gt_3k": 820,
    "m_sn_gt_4k": 465,
    "m_sn_gt_500": 951,
    "m_sn_gt_6k": 736,
    "m_sn_lt_1k": 180,
    "m_sn_lt_2k": 529,
    "m_sn_lt_3k": 543,
    "middle_wave_poor": 896,
    "mod_gt_4k": 755,
    "mod_mixed": 811,
    "vmod_s_mixed": 956,
    "mod_s_sn_gt_500": 542,
    "mod_sn": 835,
    "mod_sn_gt_1k": 814,
    "mod_sn_gt_2k": 207,
    "mod_sn_gt_3k": 166,
    "mod_sn_gt_4k": 732,
    "mod_sn_gt_500": 204,
    "notch_4k": 80,
    "notch_at_4k": 698,
    "o_ar_c": 823,
    "o_ar_u": 147,
    "s_sn_gt_1k": 577,
    "s_sn_gt_2k": 493,
    "s_sn_gt_4k": 993,
    "speech": 585,
    "static_normal": 654,
    "tymp": 677,
    "viith_nerve_signs": 657,
    "wave_V_delayed": 585,
    "waveform_ItoV_prolonged": 793,
    "p-index": 659,
    "labels": 200
}, columns=cols_audiology)

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
    labels_col = df_audiology.pop("labels")
    df_audiology.insert(0, "labels", labels_col)
    return df_audiology


# Choose dataset
dataset = load_car()
# dataset = load_mushroom()
# dataset = load_audiology()

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

print("Cols created.")

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
        """
        print(
            "Columns in column transformer: ",
            current_columns_one_hot,
            " and ",
            current_columns_ordinal,
        )
        """

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


# Sequential Forward Feature Selector
class SequentialForwardFeatureSelector:
    def __init__(self, classification_costs, CV_folds, uncertainty_threshold):
        self.classification_costs = classification_costs
        self.CV_folds = CV_folds
        self.uncertainty_threshold = uncertainty_threshold

    def sequential_predict(
            self,
            X_train_original,
            y_train_original,
            X_test_original,
            y_test_original,
            ordinal_categoires_order,
            cols_ordinal,
            cols_one_hot,
            whole_dataset,
            data_duplication_flag,
    ):
        # Make copies as we'll be altering these datasets
        X_tr = copy.deepcopy(X_train_original)
        y_tr = copy.deepcopy(y_train_original)
        X_tst = copy.deepcopy(X_test_original)
        y_tst = copy.deepcopy(y_test_original)

        # Variables
        final_result_columns = [
            "highest_proba",
            "outcome",
            "cost_of_classification",
            "feature_used_to_classify",
        ]

        total_number_of_cases = np.shape(X_tst)[0]
        classified_size = -1

        for index, row_entry in X_tst.iterrows():
            unclassified_flag = True
            duplicates = pd.DataFrame()
            final_result_dataframe = pd.DataFrame(columns=final_result_columns)
            unused_features = X_tst.columns.tolist()
            current_features = []

            # Statistics
            loop_number = 0
            classified_size += 1

            print("")
            print(
                "Classified classes: ",
                classified_size,
                "/",
                total_number_of_cases,
                " | ",
                "{:.2f}".format(classified_size / total_number_of_cases * 100),
                "%",
            )

            print("")
            print("=========================================")
            print("New row entry number:", index)

            # starting feature
            accuracy_per_new_feature = self.find_next_best_feature(
                X_tr,
                y_tr,
                None,
                None,
                unused_features,
                current_features,
                whole_dataset,
                ordinal_categoires_order,
                cols_ordinal,
                cols_one_hot,
            )
            latch_flag = False

            # Main loop, until all test classes are classified
            while unclassified_flag:
                # make a list of features with their predicted accuracy
                print("")
                print("Start of loop number: ", loop_number)
                loop_number += 1

                if latch_flag:
                    accuracy_per_new_feature = self.find_next_best_feature(
                        X_tr,
                        y_tr,
                        X_train_add_all_features,
                        y_train_add,
                        unused_features,
                        current_features,
                        whole_dataset,
                        ordinal_categoires_order,
                        cols_ordinal,
                        cols_one_hot,
                    )
                else:
                    # skips only first iteration
                    latch_flag = True

                # we have all features and their accuracies, we pick the best one
                best_next_feature = pd.DataFrame(accuracy_per_new_feature).idxmax(
                    axis=1
                )[0]
                # and adjust feature trackers
                unused_features.remove(best_next_feature)
                current_features.append(best_next_feature)

                print("Picked feature: ", best_next_feature)
                print("Current feature set: ", current_features)

                # 1.We have chosen the best feature, duplicate data and we classify the test using the feature and checking the probablilities

                # create X_train subset with apropriate features
                X_train_subset = pd.DataFrame(X_tr[current_features])
                X_test_subset = pd.DataFrame(row_entry.loc[current_features]).T

                row_features = X_test_subset.columns.tolist()
                row_features.remove(best_next_feature)
                feature_data = pd.DataFrame()
                if row_features is not None:
                    feature_data = X_test_subset[row_features]
                X_train_add_all_features, y_train_add = self.prepare_train_dataset(
                    X_tr, y_tr, feature_data, data_duplication_flag,
                )

                X_train_add = X_train_add_all_features[X_train_subset.columns.tolist()]

                X_train_dup = pd.concat([X_train_subset, X_train_add], axis=0)
                y_train_dup = np.concatenate((y_tr, y_train_add), axis=0)

                # make new classifier (due to different encoder data)

                classifier_per_featureset = self.make_encoding_categorical_bayes(
                    ordinal_categoires_order,
                    cols_ordinal,
                    cols_one_hot,
                    pd.DataFrame(dataset[current_features]),
                )

                # define duplicates before end of refactorization and moving forward
                outcomes, highest_probas, duplicates = self.predict_proba_wrapper(
                    classifier_per_featureset, X_train_dup, y_train_dup, X_test_subset
                )

                # 2.If proba > threshold, the we move/pop them from X_test to results along with the statistics

                cost_of_classification = pd.DataFrame(
                    self.get_classification_costs(current_features),
                    columns=["cost_of_classification"],
                    index=[index],
                )

                feature_used_to_classify = pd.DataFrame(
                    ",".join(map(str, current_features)),
                    columns=["feature_used_to_classify"],
                    index=[index],
                )
                # print(feature_used_to_classify)

                # remove already classified classes
                condition = highest_probas["highest_proba"] > (
                        1 - self.uncertainty_threshold
                )

                batch_result_dataframe = pd.concat(
                    [
                        outcomes,
                        highest_probas,
                        cost_of_classification,
                        feature_used_to_classify,
                    ],
                    axis=1,
                )

                rows_classified = highest_probas.loc[condition].index
                rows_unclassified = X_test_subset.index.difference(rows_classified)
                if rows_unclassified.empty or len(unused_features) == 0:
                    unclassified_flag = False
                    final_result_dataframe = pd.concat(
                        [final_result_dataframe, batch_result_dataframe]
                    )

                # now go back to the beginning of the loop and check for unclassified classes
            print(
                "Out of the loop. Usable features ran out, or no more cases to classify."
            )
        print("All cases classified.")
        return final_result_dataframe

    def predict_proba_wrapper(self, classifier, X_train, y_train, X_test):
        outcomes_fn = pd.DataFrame(columns=["outcome"])
        highest_probas_fn = pd.DataFrame(columns=["highest_proba"])
        to_duplicate_next = pd.DataFrame()

        classifier.fit(X_train, y_train)

        df_row_entry = X_test

        # gotta make it in 2 steps bc of no column name tracking in numpy
        new_outcome_df = pd.DataFrame(
            classifier.predict(df_row_entry),
            columns=["outcome"],
            index=df_row_entry.index,
        )

        outcomes_fn = pd.concat([outcomes_fn, new_outcome_df])

        probas_fn = classifier.predict_proba(df_row_entry)
        new_probas_df = pd.DataFrame(
            np.max(np.max(probas_fn, axis=1), axis=0),
            columns=["highest_proba"],
            index=df_row_entry.index,
        )
        highest_probas_fn = pd.concat([highest_probas_fn, new_probas_df])
        return outcomes_fn, highest_probas_fn, to_duplicate_next

    def prepare_train_dataset(
            self, X_train_arg, y_train_arg, duplicates_per_case, flag
    ):
        if duplicates_per_case.empty or not flag:
            # nothing to dupe or flag is down (skip)
            return X_train_arg, y_train_arg

        X_train = copy.deepcopy(X_train_arg)
        y_train = pd.DataFrame(
            copy.deepcopy(y_train_arg), columns=["labels"], index=X_train.index
        )

        # make 1 full dataset for easy modification
        full_test_data = pd.concat([X_train, y_train], axis=1)

        # for each feature
        duplicate_rows = pd.DataFrame(columns=X_train.columns.tolist())
        for col_name in duplicates_per_case.axes[1].tolist():
            dupes = full_test_data.apply(
                lambda row: row[
                    full_test_data[col_name].isin([duplicates_per_case[col_name]])
                ]
            )
            duplicate_rows = pd.concat(
                [duplicate_rows, dupes], axis=0, ignore_index=True
            )

        # return X_train, y_train
        return (
            duplicate_rows.loc[:, duplicate_rows.columns != "labels"],
            duplicate_rows.loc[:, "labels"],
        )

    def find_next_best_feature(
            self,
            X_tra,
            y_tra,
            X_train_add,
            y_train_add,
            unused_feat,
            current_feat,
            whole_dataset,
            ordinal_categoires_order,
            cols_ordinal,
            cols_one_hot,
    ):
        X_tr = copy.deepcopy(X_tra)
        y_tr = copy.deepcopy(y_tra)
        # size_beginning = np.shape(X_tr)[0]
        if X_train_add is not None and y_train_add is not None:
            X_tr = pd.concat([X_tra, X_train_add], axis=0)
            y_tr = np.concatenate((y_tra, y_train_add), axis=0)
        # print("Added: ", X_tr.shape[0] - size_beginning)
        kf = KFold(n_splits=self.CV_folds)
        accuracy_per_new_feature = pd.DataFrame(
            0, index=np.arange(1), columns=unused_feat,
        )
        for new_feature in unused_feat:
            # print("Calculating feature: ", new_feature)
            sum_of_accuracies = 0
            feature_set_to_try = copy.deepcopy(current_feat)
            feature_set_to_try.append(new_feature)
            dataset_for_encoder = pd.DataFrame(whole_dataset[feature_set_to_try])

            for train_index, test_index in kf.split(X_tr):
                # create _train, _cv_test, _test splits
                # no need to reshuffle it, it's already in random order
                # X is a dataframe
                X_train, X_cv = (
                    X_tr.iloc[train_index],
                    X_tr.iloc[test_index],
                )

                # y
                y_train, y_cv = (
                    y_tr[train_index],
                    y_tr[test_index],
                )

                # print("train: ", train_index, "test: ", test_index)

                # add feture to test to X
                X_train_subset = pd.DataFrame(X_train[feature_set_to_try])
                X_cv_subset = pd.DataFrame(X_cv[feature_set_to_try])

                # make classifier
                classifier = self.make_encoding_categorical_bayes(
                    ordinal_categoires_order,
                    cols_ordinal,
                    cols_one_hot,
                    dataset_for_encoder,
                )

                # train classifier
                classifier.fit(X_train_subset, y_train)
                # predict
                y_cv_prediciton = classifier.predict(X_cv_subset)
                # judge accuracy of new feature subset
                sum_of_accuracies += metrics.accuracy_score(y_cv, y_cv_prediciton)

            # save accuracy per new feature
            accuracy_per_new_feature[new_feature] = sum_of_accuracies / self.CV_folds

        return accuracy_per_new_feature

    def make_encoding_categorical_bayes(
            self, ordinal_categoires_order, cols_ordinal, cols_one_hot, whole_dataset
    ):
        return EncodingCategoricalBayes(
            # classifier=CategoricalNB(),
            ordinal_categories_order=order_of_ordinal_categories,
            ordinal_columns=cols_ordinal,
            one_hot_columns=cols_one_hot,
            dataset=whole_dataset,
        )

    def get_classification_costs(self, list_of_categories):
        return self.classification_costs[
            self.classification_costs.columns.intersection(list_of_categories)
        ].sum(axis=1)[0]


print("Sequential Forward Feature Selector is created.")

# Create a Bayes Classifier || requires min_categories due to a bug with indexes, reporting the bug added to TODO

selector = SequentialForwardFeatureSelector(dataset_costs, 10, 0.15)

# Train the model using the training sets
results = selector.sequential_predict(
    X_train,
    encoded_y_train,
    X_test,
    encoded_y_test,
    order_of_ordinal_categories,  # order of ordinal categories
    cols_ordinal,  # list of ordinal columns in whole data
    cols_one_hot,  # list of one hot columns in whole data
    X_cat,  # encoder dataset
    True,  # feature duplication when classifying
)

print("Done")
# print(results)

# warning: not sorted!
y_pred = results["outcome"].sort_index()
y_test.sort_index(inplace=True)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100, "%")
print("F1 score:", metrics.f1_score(y_test, y_pred, average="weighted") * 100, "%")

# Deduct some useful metrics:
# mean cost
# median cost
# difference in accuracy

# save to csv
file_name = 'results dependent feature selection'  # + proces.ppid
results.to_csv(file_name, sep='\t', encoding='utf-8')
