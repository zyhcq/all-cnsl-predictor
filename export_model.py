import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = r"d:\work_521"
PROCESSED_DIR = os.path.join(BASE_DIR, "02_processed_data")
FINAL_DIR = os.path.join(BASE_DIR, "04_final_model")
MODEL_DIR = os.path.join(BASE_DIR, "05_models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend_new")
os.makedirs(FRONTEND_DIR, exist_ok=True)

def main():
    print("Loading models...")
    cox_relapse = joblib.load(os.path.join(MODEL_DIR, "cox_relapse_spca.pkl"))
    cox_death = joblib.load(os.path.join(MODEL_DIR, "cox_death_spca.pkl"))
    
    coef_relapse = cox_relapse.params_.to_dict()
    coef_death = cox_death.params_.to_dict()
    
    norm_mean_relapse = cox_relapse._norm_mean.to_dict() if hasattr(cox_relapse, '_norm_mean') else {}
    norm_mean_death = cox_death._norm_mean.to_dict() if hasattr(cox_death, '_norm_mean') else {}
    
    # Extract baseline cumulative hazard for relapse
    bch_relapse_df = cox_relapse.baseline_cumulative_hazard_
    times_r = bch_relapse_df.index.tolist()
    hazards_r = bch_relapse_df.iloc[:, 0].tolist()
    
    # Extract baseline cumulative hazard for death
    bch_death_df = cox_death.baseline_cumulative_hazard_
    times_d = bch_death_df.index.tolist()
    hazards_d = bch_death_df.iloc[:, 0].tolist()
    
    print("Loading params...")
    with open(os.path.join(PROCESSED_DIR, "impute_params.json"), "r", encoding="utf-8") as f:
        impute_params = json.load(f)
        
    with open(os.path.join(MODEL_DIR, "spca_params.json"), "r", encoding="utf-8") as f:
        spca_params = json.load(f)
        
    # We need to extract the means of the SPCA variables from the training set,
    # because SPCA (like PCA) centers the data based on the training mean before projecting.
    # Oh wait! In SPCA, `SparsePCA`'s `transform()` method automatically centers using `pca.mean_`.
    # Let's extract `mean_` directly from `final_model_data.csv` which has the _z features.
    
    print("Extracting exact SPCA means from final_model_data.csv...")
    df_final = pd.read_csv(os.path.join(PROCESSED_DIR, "final_model_data.csv"))
    df_train = df_final[df_final["cohort_role"] == "Train_70"]
    
    for c_name, p_data in spca_params.items():
        features = p_data["features"]
        means = df_train[features].mean().tolist()
        spca_params[c_name]["mean"] = means
        
    # To compute _z scores for the UI, we need the mean and std of the raw/log features.
    # Let's compute them dynamically from df_train if they are not saved in scaler_params.json.
    # We know z = (x - mean) / std. So we need the raw mean and std.
    # Wait, the frontend needs to replicate the `z` scaling.
    # In `final_model_data.csv`, the features are ALREADY standard scaled.
    # To get the original mean and std, we need the raw data.
    print("Computing standard scaler parameters from raw data...")
    df_cleaned = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_data.csv"))
    # In the updated pipeline, how is train defined in cleaned_data.csv?
    # Usually it's 'dataset' == 'train' or 'cohort' in ['CCLG-ALL-2008', 'CCCG-ALL-2015']
    if 'dataset' in df_cleaned.columns:
        df_raw_train = df_cleaned[df_cleaned['dataset'] == 'train'].copy()
    else:
        # fallback
        df_raw_train = df_cleaned.copy()
        
    scaler_params = {}
    
    def get_stats(series):
        std = series.std()
        return {'mean': float(series.mean()), 'std': float(std if std != 0 else 1.0)}
        
    # Replicate log transforms
    log_cols = ['wbc', 'anc', 'monocyte', 'plt', 'ldh', 'uric_acid', 'ast', 'alt', 'alp',
                'csf_wbc', 'csf_protein', 'crp', 'aptt', 'd_dimer', 'pct', 'csf_rbc']
    for col in log_cols:
        if col in df_raw_train.columns:
            s = np.log1p(df_raw_train[col].clip(lower=0))
            scaler_params[f"{col}_log"] = get_stats(s)
            
    # Z-only
    z_cols = ['age_months', 'albumin', 'csf_glucose', 'pt', 'fibrinogen', 'prealbumin']
    for col in z_cols:
        if col in df_raw_train.columns:
            scaler_params[col] = get_stats(df_raw_train[col])
            
    # Derived ratios
    df_raw_train['alc'] = df_raw_train['alc'].replace(0, 0.01)
    df_raw_train['wbc'] = df_raw_train['wbc'].replace(0, 0.01)
    df_raw_train['alt'] = df_raw_train['alt'].replace(0, 0.01)
    df_raw_train['csf_protein'] = df_raw_train['csf_protein'].replace(0, 0.01)
    
    if all(x in df_raw_train.columns for x in ['anc', 'alc', 'plt', 'monocyte', 'ast', 'alt', 'ldh', 'wbc', 'albumin', 'csf_wbc', 'csf_protein']):
        scaler_params['nlr_log'] = get_stats(np.log1p(df_raw_train['anc'] / df_raw_train['alc']))
        scaler_params['plr_log'] = get_stats(np.log1p(df_raw_train['plt'] / df_raw_train['alc']))
        scaler_params['mlr_log'] = get_stats(np.log1p(df_raw_train['monocyte'] / df_raw_train['alc']))
        scaler_params['sii_log'] = get_stats(np.log1p((df_raw_train['plt'] * df_raw_train['anc']) / df_raw_train['alc']))
        scaler_params['ast_alt_log'] = get_stats(np.log1p(df_raw_train['ast'] / df_raw_train['alt']))
        scaler_params['lwr_log'] = get_stats(np.log1p(df_raw_train['ldh'] / df_raw_train['wbc']))
        scaler_params['pni_log'] = get_stats(np.log1p(df_raw_train['albumin'] + 5 * df_raw_train['alc']))
        scaler_params['csf_wbc_prot_log'] = get_stats(np.log1p(df_raw_train['csf_wbc'] / df_raw_train['csf_protein']))

    model_config = {
        "impute_params": impute_params,
        "scaler_params": scaler_params,
        "spca_params": spca_params,
        "cox_relapse": {
            "coefficients": coef_relapse,
            "norm_mean": norm_mean_relapse,
            "baseline_cumulative_hazard": {
                "times": times_r,
                "hazards": hazards_r
            }
        },
        "cox_death": {
            "coefficients": coef_death,
            "norm_mean": norm_mean_death,
            "baseline_cumulative_hazard": {
                "times": times_d,
                "hazards": hazards_d
            }
        }
    }
    
    out_js_path = os.path.join(FRONTEND_DIR, "model_config.js")
    with open(out_js_path, "w", encoding="utf-8") as f:
        f.write("const MODEL_CONFIG = ")
        json.dump(model_config, f, indent=2, ensure_ascii=False)
        f.write(";\n")
        
    print(f"Successfully exported model configuration to {out_js_path}")

if __name__ == "__main__":
    main()
