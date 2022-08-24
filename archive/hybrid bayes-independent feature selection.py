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
    # df_car = df_car.drop("buying", axis=1)
    # df_car = df_car.drop("maintenance", axis=1)
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
    # df_mushroom = df_mushroom.drop("odor", axis=1)
    # df_mushroom = df_mushroom.drop("spore_print_color", axis=1)
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
    # df_audiology = df_audiology.drop("age_gt_60", axis=1)
    labels_col = df_audiology.pop("labels")
    df_audiology.insert(0, "labels", labels_col)
    return df_audiology


# Choose dataset
dataset = load_car()
# dataset_costs = car_cost

# dataset = load_mushroom()
# dataset_costs = mushroom_cost

# dataset = load_audiology()
# dataset_costs = audiology_cost

# print(dataset.info())
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


# print("Class EncodingCategoricalBayes has been created")


# Sequential Forward Feature Selector
class SequentialForwardFeatureSelector:
    def __init__(self, CV_folds, uncertainty_threshold):
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
            data_duplication_flag
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
            "used_features"
        ]

        duplicates = pd.DataFrame()
        final_result_dataframe = pd.DataFrame(columns=final_result_columns)
        unused_features = X_tst.columns.tolist()
        current_features = []

        # Statistics
        loop_number = 0
        total_number_of_cases = np.shape(X_tst)[0]
        classified_size = 0

        # Main loop, until all test classes are classified
        while unused_features and not X_tst.empty:
            # make a list of features with their predicted accuracy
            # print("Start of loop number: ", loop_number)
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

            # we have all features and their accuracies, we pick the best one
            best_next_feature = pd.DataFrame(accuracy_per_new_feature).idxmax(axis=1)[0]
            # and adjust feature trackers
            unused_features.remove(best_next_feature)
            current_features.append(best_next_feature)

            # print("Picked feature: ", best_next_feature)
            # print("Current feature set: ", current_features)

            # 1.We have chosen the best feature and we classify the test using the feature and checking the probablilities

            # create X_train subset with apropriate features
            X_train_subset_full = pd.DataFrame(X_tr[current_features])
            X_test_subset_full = pd.DataFrame(X_tst[current_features])
            # make new classifier (due to different encoder data)
            classifier = self.make_encoding_categorical_bayes(
                ordinal_categoires_order,
                cols_ordinal,
                cols_one_hot,
                pd.DataFrame(dataset[current_features]),
            )
            # define duplicates before end of refactorization and moving forward
            outcomes, highest_probas, duplicates = self.predict_proba_wrapper(
                classifier,
                X_train_subset_full,
                y_tr,
                X_test_subset_full,
                duplicates,
                data_duplication_flag,
                best_next_feature,
            )

            # 2.If proba > threshold, then we move/pop them from X_test to results along with the statistics

            cost_of_classification = pd.DataFrame(
                self.get_classification_costs(current_features),
                columns=["cost_of_classification"],
                index=X_tst.index,
            )

            # remove already classified classes
            condition = highest_probas["highest_proba"] > (
                    1 - self.uncertainty_threshold
            )

            iter_features = pd.DataFrame(",".join(current_features),
                                         columns=["used_features"],
                                         index=X_tst.index)

            batch_result_dataframe = pd.concat(
                [
                    outcomes,
                    highest_probas,
                    cost_of_classification,
                    iter_features
                ],
                axis=1
            )

            # print("Final result dataframe:")
            # print(batch_result_dataframe)
            # print("Are X test and result dataframe indexes equal? :",X_tst.index.equals(batch_result_dataframe.index))

            rows_classified = highest_probas.loc[condition].index
            rows_unclassified = X_tst.index.difference(rows_classified)
            y_tst = pd.DataFrame(y_tst, columns=["labels"], index=X_tst.index)

            # print("Rows classified: ", rows_classified)
            # print("Rows unclassified: ", rows_unclassified)
            # print("X test before:")
            # print(X_tst)
            # print("Number of rows to drop:", np.shape(rows_classified)[0])
            # print("Number of rows to leave:", np.shape(rows_unclassified)[0])

            X_tst.drop(rows_classified, inplace=True)
            y_tst.drop(rows_classified, inplace=True)
            # if unused features is empty, append everything -> if unused features is not empty, drop unneeded
            if unused_features:
                batch_result_dataframe.drop(rows_unclassified, inplace=True)
            # print("Are X test and result dataframe indexes unequal? :",not X_tst.index.equals(batch_result_dataframe.index),)

            # print("final: ")
            # print(batch_result_dataframe)
            final_result_dataframe = pd.concat(
                [final_result_dataframe, batch_result_dataframe]
            )

            # percentage classified
            classified_size += rows_classified.size
            # print("End of loop ", loop_number)
            loop_number += 1
            """
            print(
                "Classified classes: ",
                classified_size,
                "/",
                total_number_of_cases,
                " | ",
                "{:.2f}".format(classified_size / total_number_of_cases * 100),
                "%",
            )
            """
            # now go back to the beginning of the loop and check for unclassified classes

        # print("Out of the loop. Usable features ran out, or no more cases to classify.")
        # results_dataframe = pd.concat([results_dataframe, final_result_dataframe], axis=0)
        return final_result_dataframe

    def predict_proba_wrapper(
            self, classifier, X_train, y_train, X_test, duplicates, flag, unknown_feature
    ):
        outcomes_fn = pd.DataFrame(columns=["outcome"])
        highest_probas_fn = pd.DataFrame(columns=["highest_proba"])
        to_duplicate_next = pd.DataFrame()
        for index, row_entry in X_test.iterrows():
            row_features = row_entry.axes[0].tolist()
            row_features.remove(unknown_feature)
            feature_data = pd.DataFrame()
            if row_features is not None:
                feature_data = row_entry.loc[row_features]
            X_train_mod, y_train_mod = self.prepare_train_dataset(
                X_train, y_train, feature_data, flag,
            )
            classifier.fit(X_train_mod, y_train_mod)

            df_row_entry = row_entry.to_frame().T

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
            # print("Nothing to duplicate")
            return X_train_arg, y_train_arg

        X_train = copy.deepcopy(X_train_arg)
        y_train = pd.DataFrame(
            copy.deepcopy(y_train_arg), columns=["labels"], index=X_train.index
        )

        # make 1 full dataset for easy modification
        full_test_data = pd.concat([X_train, y_train], axis=1)
        bef = np.shape(full_test_data)[0]
        # print("Before duplication:", bef)

        # for each feature
        duplicate_rows = pd.DataFrame(columns=X_train.columns.tolist())
        for col_name in duplicates_per_case.axes[0].tolist():
            dupes = full_test_data.apply(
                lambda row: row[
                    full_test_data[col_name].isin([duplicates_per_case[col_name]])
                ]
            )
            duplicate_rows = pd.concat(
                [duplicate_rows, dupes], axis=0, ignore_index=True
            )
        full_test_data = pd.concat(
            [full_test_data, duplicate_rows], axis=0, ignore_index=True
        )
        # print("After duplication:", np.shape(full_test_data)[0])
        # print("Appended", str(np.shape(full_test_data)[0] - bef), " examples.")

        # print("Delta:", np.shape(full_test_data)[0])

        # return X_train, y_train
        return (
            full_test_data.loc[:, full_test_data.columns != "labels"],
            full_test_data.loc[:, "labels"],
        )

    def find_next_best_feature(
            self,
            X_tra,
            y_tra,
            X_test,
            y_test,
            unused_feat,
            current_feat,
            whole_dataset,
            ordinal_categoires_order,
            cols_ordinal,
            cols_one_hot,
    ):
        X_tr = copy.deepcopy(X_tra)
        y_tr = copy.deepcopy(y_tra)
        kf = StratifiedKFold(n_splits=self.CV_folds)
        accuracy_per_new_feature = pd.DataFrame(
            0, index=np.arange(1), columns=unused_feat,
        )
        for new_feature in unused_feat:
            # print("Calculating feature: ", new_feature)
            sum_of_accuracies = 0
            feature_set_to_try = copy.deepcopy(current_feat)
            feature_set_to_try.append(new_feature)
            dataset_for_encoder = pd.DataFrame(whole_dataset[feature_set_to_try])

            for train_index, test_index in kf.split(X_tr, y_tr):
                # create _train, _cv_test, _test splits
                # no need to reshuffle it, it's already in random order
                # X is a dataframe
                X_train, X_cv = (
                    X_tr.iloc[train_index],
                    X_tr.iloc[test_index],
                )

                # y is a numpy array
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
            # print("Finished calculations for feature: ",new_feature,"Accuracy: ",accuracy_per_new_feature[new_feature][0])

        return accuracy_per_new_feature

    def make_encoding_categorical_bayes(
            self, ordinal_categoires_order, cols_ordinal, cols_one_hot, whole_dataset
    ):
        return EncodingCategoricalBayes(
            # classifier=CategoricalNB(),
            ordinal_categories_order=ordinal_categoires_order,
            ordinal_columns=cols_ordinal,
            one_hot_columns=cols_one_hot,
            dataset=whole_dataset,
        )

    def get_classification_costs(self, list_of_categories):
        return len(list_of_categories)


# print("Sequential Forward Feature Selector is created.")

# Create a Bayes Classifier || requires min_categories due to a bug with indexes, reporting the bug added to TODO

def doClassification(iteration_id, number_of_folds, threshold, X_train, encoded_y_train, X_test, encoded_y_test,
                     full_encoder_dataset, data_dupl_flag):
    selector = SequentialForwardFeatureSelector(number_of_folds, threshold)

    # Train the model using the training sets
    results = selector.sequential_predict(
        X_train,
        encoded_y_train,
        X_test,
        encoded_y_test,
        order_of_ordinal_categories,  # order of ordinal categories
        cols_ordinal,  # list of ordinal columns in whole data
        cols_one_hot,  # list of one hot columns in whole data
        full_encoder_dataset,  # encoder dataset
        data_dupl_flag,  # feature duplication when classifying
    )

    results.sort_index(inplace=True)
    file_name = 'results_independent_feature_selection_' + str(iteration_id) + '_duplication_' + str(
        data_dupl_flag) + '.csv'
    results.to_csv(file_name, sep='\t', encoding='utf-8', mode='w', header=True, index=True)

    # warning: not sorted!
    y_pred = results["outcome"].sort_index().astype(dtype='int32')
    results.sort_index(inplace=True)
    f1_acc_res = pd.DataFrame([(
        metrics.accuracy_score(encoded_y_test, y_pred),
        metrics.f1_score(encoded_y_test, y_pred, average="weighted"))],
        columns=["accuracy", "F1"], index=[0])
    print("Accuracy of " + str(iteration_id) + " fold:", metrics.accuracy_score(encoded_y_test, y_pred) * 100, "%")
    print("F1 score of " + str(iteration_id) + " fold:",
          metrics.f1_score(encoded_y_test, y_pred, average="weighted") * 100, "%")


if __name__ == '__main__':
    number_of_processes = 10
    processes = []

    max_seed_val = 2 ** 32 - 1
    random_seed_kfold = random.randrange(0, max_seed_val)
    threshold = 0.15
    data_dupl_flag = True
    le = LabelEncoder().fit(y_cat)
    reference_outcomes = le.transform(y_cat)

    skf = StratifiedKFold(n_splits=number_of_processes, random_state=random_seed_kfold, shuffle=True)
    proc = 0
    # Split dataset into training set and test set
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

        next_proc = copy.deepcopy(proc)
        proc += 1
        p = Process(target=doClassification,
                    args=(next_proc, number_of_processes, threshold, X_train, encoded_y_train, X_test,
                          encoded_y_test, X_cat, data_dupl_flag))
        processes.append(p)
        p.start()

    for pr in processes:
        pr.join()

    combined_results = pd.DataFrame(columns=[
        "index",
        "highest_proba",
        "outcome",
        "cost_of_classification",
        "used_features"
    ])
    # combined_results_acc = pd.DataFrame(columns=["index", "accuracy", "F1"])

    for i in range(number_of_processes):
        current_file_name = 'results_independent_feature_selection_' + str(i) + '_duplication_' + str(
            data_dupl_flag) + '.csv'
        combined_results = pd.concat(
            [combined_results, pd.read_csv(current_file_name,
                                           names=["index", "highest_proba", "outcome", "cost_of_classification",
                                                  "used_features"],
                                           sep="\t", header=0)], axis=0)
        os.remove(current_file_name)

    combined_results.sort_values(by=['index'], inplace=True)
    # combined_results_acc.sort_values(by=['index'], inplace=True)
    all_outcomes = combined_results["outcome"].astype(dtype='int32')
    reference_outcomes = le.transform(y_cat)

    combined_results_acc = pd.DataFrame([(
        metrics.accuracy_score(reference_outcomes, all_outcomes),
        metrics.f1_score(reference_outcomes, all_outcomes, average="weighted"))],
        columns=["accuracy", "F1"], index=[0])
    print("Total accuracy:", metrics.accuracy_score(reference_outcomes, all_outcomes) * 100, "%")
    print("Total F1 score:", metrics.f1_score(reference_outcomes, all_outcomes, average="weighted") * 100, "%")

    combined_results.to_csv('results_independent_feature_selection_w_duplication_' + str(data_dupl_flag) + '.csv',
                            sep='\t', encoding='utf-8',
                            mode='a',
                            header=True, index=False)
    combined_results_acc.to_csv(
        'results_independent_feature_selection_w_duplication_acc_f1_' + str(data_dupl_flag) + '.csv',
        sep='\t',
        encoding='utf-8', mode='a',
        header=True, index=False)
