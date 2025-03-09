import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

trial_pp_dir = "trial_pp"
trial_itt_dir = "trial_itt"

os.makedirs(trial_pp_dir, exist_ok=True)
os.makedirs(trial_itt_dir, exist_ok=True)

# Load dataset
data_censored = pd.read_csv("data_censored.csv")

# Add missing columns to avoid KeyError in Step 5
data_censored["assigned_treatment"] = data_censored["treatment"]  # Placeholder for treatment assignment
data_censored["followup_time"] = data_censored["period"]  # Initialize follow-up time

print(data_censored.head())

# Logistic Regression for treatment switching
switch_model = LogisticRegression()
X_switch = data_censored[["age", "x1", "x3"]]
y_switch = data_censored["treatment"]

switch_model.fit(X_switch, y_switch)
data_censored["switch_weight"] = switch_model.predict_proba(X_switch)[:, 1]

censor_model = LogisticRegression()
X_censor = data_censored[["x2", "x1"]]
y_censor = 1 - data_censored["censored"]

censor_model.fit(X_censor, y_censor)
data_censored["censor_weight"] = censor_model.predict_proba(X_censor)[:, 1]

data_censored["final_weight"] = data_censored["switch_weight"] * data_censored["censor_weight"]
print(data_censored[["id", "final_weight"]].head())

X_outcome = sm.add_constant(data_censored[["assigned_treatment", "x2", "followup_time"]])
y_outcome = data_censored["outcome"]

outcome_model = sm.Logit(y_outcome, X_outcome).fit()
print(outcome_model.summary())

def expand_trials(data):
    expanded_data = data.copy()
    expanded_data["trial_period"] = expanded_data["period"]
    expanded_data["followup_time"] = expanded_data["period"] + 1
    return expanded_data

trial_pp_expanded = expand_trials(data_censored)
trial_itt_expanded = expand_trials(data_censored)

print(trial_itt_expanded.head())


def create_sequence_of_trials(data):
    expanded_trials = []
    
    for _, row in data.iterrows():
        period_start = int(row["period"])  # Ensure it's an integer
        period_end = period_start + 5  # Expanding up to 5 periods ahead
        
        for t in range(period_start, period_end):
            new_row = row.copy()
            new_row["trial_period"] = t
            new_row["followup_time"] = t - period_start
            expanded_trials.append(new_row)
    
    return pd.DataFrame(expanded_trials)

# Apply to expanded data
trial_pp_seq = create_sequence_of_trials(trial_pp_expanded)
trial_itt_seq = create_sequence_of_trials(trial_itt_expanded)

print(trial_itt_seq.head())


sample_size = int(len(trial_itt_expanded) * 0.5)  # 50% control group
trial_itt_sampled = trial_itt_expanded.sample(sample_size, random_state=1234)
print(trial_itt_sampled.head())


def winsorize_weights(weights):
    q99 = np.percentile(weights, 99)
    return np.minimum(weights, q99)

trial_itt_sampled["adjusted_weight"] = winsorize_weights(trial_itt_sampled["final_weight"])

X_msm = sm.add_constant(trial_itt_sampled[["assigned_treatment", "x2", "followup_time"]])
y_msm = trial_itt_sampled["outcome"]

msm_model = sm.Logit(y_msm, X_msm, weights=trial_itt_sampled["adjusted_weight"]).fit()
print(msm_model.summary())

import matplotlib.pyplot as plt

followup_times = np.arange(0, 11)
survival_probs = msm_model.predict(sm.add_constant(trial_itt_sampled[["assigned_treatment", "x2", "followup_time"]]))

plt.plot(followup_times, survival_probs[:len(followup_times)], label="Survival Difference")
plt.xlabel("Follow-up Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.show()