import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

datapath = "..\\results\\sequential with data duplication\\modified datasets\\results_dependent_feature_selection_mushroom_False_modified_odor_spore_print_color.csv"

results = df = pd.read_csv(datapath,
                 header=0,
                 sep='\t',
                 names=["highest_proba", "outcome", "cost_of_classification",
                        "used_features"])

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

def load_audiology():
    df_audiology = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.data",
        header=None,
        names=cols_audiology,
    )
    df_audiology = df_audiology.drop("p_index", axis=1)
    df_audiology = df_audiology.drop("age_gt_60", axis=1)
    df_audiology = df_audiology.drop("speech", axis=1)
    labels_col = df_audiology.pop("labels")
    df_audiology.insert(0, "labels", labels_col)
    return df_audiology

refdata = load_audiology()

le = LabelEncoder().fit(refdata["labels"])

print(">>np.column_stack(np.unique(le.inverse_transform(results['outcome']), return_counts=True))")
print(np.column_stack(np.unique(le.inverse_transform(results['outcome']), return_counts=True)))
