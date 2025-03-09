# 1. SETUP
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Define directories for output (similar to tempdir() in R)
trial_pp_dir = "trial_pp"
trial_itt_dir = "trial_itt"

os.makedirs(trial_pp_dir, exist_ok=True)
os.makedirs(trial_itt_dir, exist_ok=True)


# 2. DATA PREPARATION - Load dataset
data_censored = pd.read_csv("data_censored.csv")

# Ensure required columns exist
required_columns = ["id", "period", "treatment", "outcome", "eligible", "censored", "x1", "x2", "x3", "age"]
missing_columns = [col for col in required_columns if col not in data_censored.columns]

if missing_columns:
    raise KeyError(f"Missing columns in data_censored: {missing_columns}")

# Add necessary columns
data_censored["assigned_treatment"] = data_censored["treatment"]
data_censored["followup_time"] = data_censored["period"]

# Create trial objects as dictionaries
trial_pp = {"data": data_censored.copy()}
trial_itt = {"data": data_censored.copy()}

print(data_censored.head())  # Verify dataset is correctly loaded


# Logistic Regression for treatment switching
switch_model = LogisticRegression()
X_switch = data_censored[["age", "x1", "x3"]]
y_switch = data_censored["treatment"]

switch_model.fit(X_switch, y_switch)
data_censored["switch_weight"] = switch_model.predict_proba(X_switch)[:, 1]

# Logistic Regression for censoring
censor_model = LogisticRegression()
X_censor = data_censored[["x2", "x1"]]
y_censor = 1 - data_censored["censored"]

censor_model.fit(X_censor, y_censor)
data_censored["censor_weight"] = censor_model.predict_proba(X_censor)[:, 1]

# 4. CALCULATE WEIGHTS

# Compute final weight
data_censored["final_weight"] = data_censored["switch_weight"] * data_censored["censor_weight"]
data_censored["final_weight"] = data_censored["final_weight"].replace(0, 1).fillna(1)

# Merge weights into expanded dataset
trial_itt_expanded = trial_itt["data"].copy()
trial_itt_expanded = trial_itt_expanded.merge(
    data_censored[["id", "final_weight"]], on="id", how="left"
)

# Assign weight column safely
trial_itt_expanded["weight"] = trial_itt_expanded["final_weight"].fillna(1)
trial_itt_expanded.drop(columns=["final_weight"], inplace=True)

print(trial_itt_expanded.shape)  # Check size after merge



# 5. SPECIFY OUTCOME MODEL
X_outcome = sm.add_constant(data_censored[["assigned_treatment", "x2", "followup_time"]])
y_outcome = data_censored["outcome"]

outcome_model = sm.Logit(y_outcome, X_outcome).fit()
print(outcome_model.summary())



# 6. EXPAND TRIALS
def create_sequence_of_trials(data, max_periods=3):
    expanded_trials = []
    
    for _, row in data.iterrows():
        period_start = int(row["period"])  
        period_end = min(period_start + max_periods, period_start + 10)

        for t in range(period_start, period_end):
            new_row = row.copy()
            new_row["trial_period"] = t
            new_row["followup_time"] = t - period_start
            expanded_trials.append(new_row)

    return pd.DataFrame(expanded_trials)

trial_itt_expanded = create_sequence_of_trials(trial_itt["data"])

print(trial_itt_expanded.head())  # Check if expansion works correctly



# 6.1 RE-MERGE WEIGHT INTO EXPANDED TRIAL DATA
trial_itt_expanded = trial_itt_expanded.merge(
    data_censored[["id", "final_weight"]].rename(columns={"final_weight": "weight"}), 
    on="id", 
    how="left"
)

# Ensure weight column exists
trial_itt_expanded["weight"] = trial_itt_expanded["weight"].fillna(1)

print(trial_itt_expanded.head())  # ✅ THIS WILL NOW WORK FOR REAL



# 7. LOAD OR SAMPLE FROM EXPANDED DATA
sample_size = min(int(len(trial_itt_expanded) * 0.5), len(trial_itt_expanded))

trial_itt_sampled = trial_itt_expanded[
    ["id", "assigned_treatment", "x2", "followup_time", "weight", "outcome"]
].sample(sample_size, random_state=1234).copy()

print(trial_itt_sampled.head())  # Verify columns exist



# 8. FIT MARGINAL STRUCTURAL MODEL

def winsorize_weights(weights):
    q99 = np.percentile(weights, 99)
    return np.minimum(weights, q99)

trial_itt_sampled["adjusted_weight"] = winsorize_weights(trial_itt_sampled["weight"])

X_msm = sm.add_constant(trial_itt_sampled[["assigned_treatment", "x2", "followup_time"]])
y_msm = trial_itt_sampled["outcome"]

msm_model = sm.Logit(y_msm, X_msm).fit()
print(msm_model.summary())




# 9. INFERENCE - Predict Survival Differences

predict_times = np.arange(0, 11)
new_data = trial_itt_sampled.copy()

# Ensure constant column exists
new_data = sm.add_constant(new_data[["assigned_treatment", "x2", "followup_time"]])

predictions = []
for t in predict_times:
    new_data["followup_time"] = t
    pred_prob = msm_model.predict(new_data)
    predictions.append(pred_prob.mean())

predictions = np.array(predictions)
std_error = np.std(predictions) / np.sqrt(len(predictions))
ci_lower = predictions - 1.96 * std_error
ci_upper = predictions + 1.96 * std_error

plt.plot(predict_times, predictions, label="Survival Difference", color="blue")
plt.fill_between(predict_times, ci_lower, ci_upper, color="red", alpha=0.3, label="95% CI")

plt.xlabel("Follow-up Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.title("Survival Difference Over Time")

plt.show()
