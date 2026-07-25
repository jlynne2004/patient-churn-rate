# Synthetic Patient Dataset for Predicting Patient Churn Rate

import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'PatientID': 'P' + pd.RangeIndex(1, n + 1).astype(str),
    'Age': np.random.randint(18, 90, size=n),
    'Gender': np.random.choice(['Male', 'Female'], size=n),
    'State': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'NJ', 'GA', 'DE', 'MD'], size=n),
    'Tenure_Months': np.random.randint(1, 120, size=n),
    'Specialty': np.random.choice(['Cardiology', 'Dermatology', 'Neurology', 'Oncology', 'Pediatrics', 'Psychiatry', 'Family Medicine'], size=n),  
    'Insurance_Type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Self-Pay'], size=n),
    'Visits_Last_Year': np.random.randint(0, 15, size=n),
    'Missed_Appointments': np.random.randint(0, 5, size=n),
    'Days_Since_Last_Visit': np.random.randint(0, 730, size=n),
    'Last_Interaction_Date': pd.to_datetime('2024-01-21') - pd.to_timedelta(np.random.randint(0, 730, size=n), unit='d'),
    'Overall_Satisfaction': np.random.randint(1, 6, size=n).astype(float),
    'Wait_Time_Satisfaction': np.random.randint(1, 6, size=n).astype(float),
    'Staff_Satisfaction': np.random.randint(1, 6, size=n).astype(float),
    'Provider_Rating': np.random.randint(1, 6, size=n).astype(float),
    'Avg_Out_Of_Pocket_Cost': np.random.uniform(20, 2000, size=n).round(2),
    'Billing_Issues': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),  # 15% chance of billing issues
    'Portal_Usage': np.random.choice([0, 1], size=n, p=[0.3, 0.7]),  # 70% chance of using the portal
    'Referrals_Made': np.random.randint(0, 5, size=n),
    'Distance_To_Facility_In_Miles': np.random.uniform(0.5, 50, size=n).round(2),
    'Churned': np.random.choice([0, 1], size=n, p=[0.8, 0.2])  # 20% chance of churn
})

# Logistic turn signal: Distance to Facility, Billing Issues, Insurance Type, and Overall Satisfaction are strong predictors of churn.
logit = (
    -2.0
    + 0.35 * (df['Distance_To_Facility_In_Miles'] > 20)
    + 2.20 * (df['Billing_Issues'] == 1).astype(float)
    + 0.8 * (df['Insurance_Type'] == 'Self-Pay').astype(float)
    - 0.03 * df['Overall_Satisfaction']
    + np.random.normal(0, 0.5, size=n) # residual noise
)

churn_prob = 1 / (1 + np.exp(-logit))
print(churn_prob.mean())  # Check the average churn probability
df['Churned'] = np.random.binomial(1, churn_prob)
print(df['Churned'].mean())  # Check the actual churn rate in the dataset

df.to_csv('raw/patient_churn_dataset.csv', index=False)