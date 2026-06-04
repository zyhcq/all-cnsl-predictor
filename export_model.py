import os
import json
import joblib
import pandas as pd

BASE_DIR = r"d:\work_521"
PROCESSED_DIR = os.path.join(BASE_DIR, "02_processed_data")
FINAL_DIR = os.path.join(BASE_DIR, "04_final_model")
MODEL_DIR = os.path.join(BASE_DIR, "05_models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

def main():
    print("Loading Cox model...")
    cox = joblib.load(os.path.join(MODEL_DIR, "cox_final.pkl"))
    
    # Extract coefficients
    coef_dict = cox.params_.to_dict()
    
    # Extract baseline survival
    baseline_surv_df = cox.baseline_survival_
    baseline_times = baseline_surv_df.index.tolist()
    baseline_probs = baseline_surv_df.iloc[:, 0].tolist()
    
    # Load JSON configs
    print("Loading preprocessing params...")
    with open(os.path.join(PROCESSED_DIR, "impute_params.json"), "r", encoding="utf-8") as f:
        impute_params = json.load(f)
        
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "r", encoding="utf-8") as f:
        scaler_params = json.load(f)
        
    with open(os.path.join(PROCESSED_DIR, "pca_params.json"), "r", encoding="utf-8") as f:
        pca_params = json.load(f)
        
    # --- NEW: Extract PCA mean_ from training set ---
    print("Extracting exact PCA mean_ values from processed_data_final.csv...")
    df_final = pd.read_csv(os.path.join(FINAL_DIR, "processed_data_final.csv"))
    df_train = df_final[df_final["dataset"] == "train"]
    
    for c_name, p_data in pca_params.items():
        features = p_data["features"]
        # Calculate exact mean for the training set (what PCA fits)
        means = df_train[features].mean().tolist()
        pca_params[c_name]["mean"] = means

    # Combine all
    model_config = {
        "impute_params": impute_params,
        "scaler_params": scaler_params,
        "pca_params": pca_params,
        "cox_params": {
            "coefficients": coef_dict,
            "baseline_survival": {
                "times": baseline_times,
                "probabilities": baseline_probs
            }
        }
    }
    
    # Write to JS file
    out_js_path = os.path.join(FRONTEND_DIR, "model_config.js")
    with open(out_js_path, "w", encoding="utf-8") as f:
        f.write("const MODEL_CONFIG = ")
        json.dump(model_config, f, indent=2, ensure_ascii=False)
        f.write(";\n")
        
    print(f"Successfully exported model configuration to {out_js_path}")

if __name__ == "__main__":
    main()
