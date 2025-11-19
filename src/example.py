##############################################################################
# Copyright (C) 2024                                                         #
#                                                                            #
# CC BY-NC-SA 4.0                                                            #
#                                                                            #
# Canonical URL https://creativecommons.org/licenses/by-nc-sa/4.0/           #
# Attribution-NonCommercial-ShareAlike 4.0 International CC BY-NC-SA 4.0     #
#                                                                            #
# Prof. Elaine Cecilia Gatto | Prof. Ricardo Cerri | Prof. Mauri Ferrandin   #
#                                                                            #
# Federal University of São Carlos - UFSCar - https://www2.ufscar.br         #
# Campus São Carlos - Computer Department - DC - https://site.dc.ufscar.br   #
# Post Graduate Program in Computer Science - PPGCC                          # 
# http://ppgcc.dc.ufscar.br - Bioinformatics and Machine Learning Group      #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
# You are free to:                                                           #
#     Share — copy and redistribute the material in any medium or format     #
#     Adapt — remix, transform, and build upon the material                  #
#     The licensor cannot revoke these freedoms as long as you follow the    #
#       license terms.                                                       #
#                                                                            #
# Under the following terms:                                                 #
#   Attribution — You must give appropriate credit , provide a link to the   #
#     license, and indicate if changes were made . You may do so in any      #
#     reasonable manner, but not in any way that suggests the licensor       #
#     endorses you or your use.                                              #
#   NonCommercial — You may not use the material for commercial purposes     #
#   ShareAlike — If you remix, transform, or build upon the material, you    #
#     must distribute your contributions under the same license as the       #
#     original.                                                              #
#   No additional restrictions — You may not apply legal terms or            #
#     technological measures that legally restrict others from doing         #
#     anything the license permits.                                          #
#                                                                            #
##############################################################################


import sys
import platform
import os
import ml_auprc_roc as ml
import numpy as np
import pandas as pd


# ----------------------------------------------- #
#    EXAMPLE FOR 1 EXPERIMENT                     #
# ----------------------------------------------- #

true = pd.read_csv("~/MultiLabelAuprcRocCurves/data/Split-1/y_true.csv")
pred = pd.read_csv("~/MultiLabelAuprcRocCurves/data/Split-1/y_proba.csv")

fpr_macro, tpr_macro, macro_auc, macro_auc_interp, macro_df = ml.robust_macro_roc_auc(true, pred, verbose=False)
print(f"Macro AUC (mean): {macro_auc:.4f}")
print(f"Macro AUC (interpolated): {macro_auc_interp:.4f}")
print(macro_df[["Label", "AUC"]])
print(fpr_macro)
print(tpr_macro)


fpr_micro, tpr_micro, auc_micro = ml.robust_micro_roc_auc(true, pred, verbose=False)
print(f"Micro-average AUC: {auc_micro:.4f}")
print(fpr_micro)
print(tpr_micro)


fpr_w, tpr_w, auc_weighted, auc_df = ml.robust_weighted_roc_auc(true, pred, verbose=False)
print(f"Weighted ROC-AUC: {auc_weighted:.4f}")
print(auc_df[["Label", "AUC", "Support", "Weight"]])


sample_auc_df, samples_auc_mean = ml.robust_sample_roc_auc(true, pred, verbose=False)
print(f"Samples average AUC = {samples_auc_mean:.4f}")
print(sample_auc_df[["Sample", "AUC"]])

ml.plot_macro_roc(
    fpr_macro, tpr_macro,
    macro_auc, macro_auc_interp, macro_df,
    show = True
)

ml.plot_micro_roc(
    fpr_micro, tpr_micro, auc_micro, 
        show = True
)

ml.plot_weighted_roc(
    fpr_w, tpr_w, auc_weighted, auc_df,
    show = True
)

ml.plot_samples_auc(
    sample_auc_df,
    show = True
)

# ----------------------------------------------- #
#    EXAMPLE FOR 10-FOLDS CROSS VALIDATION        #
# ----------------------------------------------- #

# folders
base_path = os.path.expanduser("~/MultiLabelAuprcRocCurves")
data_path = os.path.join(base_path, "data")
plots_path = os.path.join(base_path, "plots")
os.makedirs(plots_path, exist_ok=True)

# df to save results
results_macro, results_micro, results_weighted, results_samples = [], [], [], []

# Loop 10 folds
for fold in range(1, 11):
    print(f"\n=================== Fold {fold} ===================")

    # Lê os arquivos do fold atual
    true = pd.read_csv(os.path.join(data_path, f"Split-{fold}/y_true.csv"))
    pred = pd.read_csv(os.path.join(data_path, f"Split-{fold}/y_proba.csv"))

    # === Macro ===
    fpr_macro, tpr_macro, macro_auc, macro_auc_interp, macro_df = ml.robust_macro_roc_auc(true, pred, verbose=True)
    results_macro.append([fold, macro_auc, macro_auc_interp])
    ml.plot_macro_roc(
        fpr_macro, tpr_macro, macro_auc, macro_auc_interp, macro_df,
        save_path=os.path.join(plots_path, f"macro_fold{fold}.pdf")
    )

    # === Micro ===
    fpr_micro, tpr_micro, auc_micro = ml.robust_micro_roc_auc(true, pred, verbose=True)
    results_micro.append([fold, auc_micro])
    ml.plot_micro_roc(
        fpr_micro, tpr_micro, auc_micro,
        save_path=os.path.join(plots_path, f"micro_fold{fold}.pdf")
    )

    # === Weighted ===
    fpr_w, tpr_w, auc_weighted, auc_df = ml.robust_weighted_roc_auc(true, pred, verbose=True)
    results_weighted.append([fold, auc_weighted])
    ml.plot_weighted_roc(
        fpr_w, tpr_w, auc_weighted, auc_df,
        save_path=os.path.join(plots_path, f"weighted_fold{fold}.pdf")
    )

    # === Samples ===
    sample_auc_df, samples_auc_mean = ml.robust_sample_roc_auc(true, pred, verbose=True)
    results_samples.append([fold, samples_auc_mean])
    ml.plot_samples_auc(
        sample_auc_df,
        save_path=os.path.join(plots_path, f"samples_fold{fold}.pdf")
    )


# =================== FINAL RESULTS ===================
df_macro = pd.DataFrame(results_macro, columns=["Fold", "Macro_AUC", "Macro_AUC_Interp"])
df_micro = pd.DataFrame(results_micro, columns=["Fold", "Micro_AUC"])
df_weighted = pd.DataFrame(results_weighted, columns=["Fold", "Weighted_AUC"])
df_samples = pd.DataFrame(results_samples, columns=["Fold", "Samples_AUC"])


# =================== SUMMARY FINAL ===================
summary_path = os.path.join(base_path, "plots/summary_roc_auc_results.csv")
summary = (
    df_macro.merge(df_micro, on="Fold")
    .merge(df_weighted, on="Fold")
    .merge(df_samples, on="Fold")
)
summary.to_csv(summary_path, index=False)

print("\nCross-validation concluída!")
print(summary)
print(f"\nResults saved in: {summary_path}")




