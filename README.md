# Creating a Patient Churn Prediction Model
This project was inspired by Karina Samsonova's latest tutorial on her website: <a href="https://karina-datascientist.beehiiv.com/p/let-s-build-a-churn-prediction-model-together">building a churn prediction model.</a> Instead of following it exactly, I'm going a little off script and building a patient churn rate prediction model instead of customer.

I initially used the patient_churn_dataset.csv provided here: https://www.kaggle.com/datasets/nudratabbas/patient-churn-prediction-dataset-for-healthcare, but the churn rate for that dataset was ~68% , well above a typical patient attrition benchmarks (~15-30% annually), and critically, no individual feature showed a meaningful correlation or statistically significant association with the Churned column. This suggested the labels may not have reflected a genuine underlying pattern. I recreated the synthetic dataset (**no real patient information is used**) using the exact same columns with a tighter control over the churn rate probability and feature importance.

## Overview
2,000 synthetic patient records, goal is to predict patient churn and quantify revenue at risk to prioritize retention outreach for healthcare practice.

## Approach
- exploratory analysis via correlation and chi-square testing across numeric and categorical features
- Random Forest ML model with balanced class weights
- threshold tuned to 0.30 (rather than default 0.5) to prioritize recall on the Churned class, since missing an at-risk patient is more costly than an unnecessary outreach
- feature importance validated using two methods (MDI and permutation importance with ROC-AUC scoring) after discovering MDI bias toward high-cardinality noise features
- revenue-at-risk layer added using Avg_Out_Of_Pocket_Cost X Visits_Last_Year X Churn Probability to prioritize outreach by dollar amount, not just risk likelihood

## Findings
23% overall churn rate (calibrated during synthetic data generation to reflect realistic patient attrition benchmarks). Billing issues are the strongest indicator of churn, followed by distance to facility, insurance type, and overall satisfaction. Classification report reflects a 0.30 decision threshold (see **Approach** for rationale).

**Classification Report:**
 | | precision | recall | f1-score | support |
 | :--- | :---: | :---: | :---: | ---: |
 | Stayed | 0.82 | 0.86 | 0.84 | 308 |
 | Churned | 0.45 | 0.38 | 0.41 | 92 |
 | accuracy | | | 0.75 | 400 |
 | macro avg | 0.64 | 0.62 | 0.63 | 400 |
 | weighted avg | 0.74 | 0.75 | 0.74 | 400 |

<img width="563" height="457" alt="image" src="https://github.com/user-attachments/assets/ee56b993-09bb-42c7-b05c-46080a9785d0" />

**Total Monthly Revenue at Risk:** $52,015.36

**Top 20 Patients to Contact First:**
| | Churn_Probability | Monthly_Revenue | Revenue_at_Risk |
| :--- | :---: | :---: | ---: |
| 1984 | 0.61 | 1916.378333 | 1168.990783 |
| 248 | 0.61 | 1537.873333 | 938.102733 |
| 1828 | 0.61 | 1350.265000 | 823.661650 |
| 320 | 0.39 | 1900.930000 | 741.362700 |
| 1724 | 0.50 | 1434.323333 | 717.161667 |
| 1922 | 0.31 | 1988.140000 | 616.323400 |
| 268 | 0.47 | 1249.606667 | 587.315133 |
| 1091 | 0.60 | 965.323333 | 579.194000 |
| 1836 | 0.26 | 2132.865000 | 554.544900 |
| 15 | 0.44 | 1258.840000 | 553.889600 |
| 1124 | 0.61 | 894.160000 | 545.437600 |
| 1916 | 0.62 | 872.316667 | 540.836333 |
| 720 | 0.48 | 1120.038333 | 537.618400 |
| 96 | 0.66 | 809.715833 | 534.412450 |
| 132 | 0.56 | 945.070000 | 529.239200 |
| 1849 | 0.35 | 1489.216667 | 521.225833 |
| 1499 | 0.32 | 1624.391667 | 519.805333 |
| 836 | 0.24 | 2063.215000 | 495.171600 |
| 1706 | 0.43 | 1104.794167 | 475.061492 |
| 1109 | 0.53 | 879.100000 | 465.923000 |

Prioritization is driven by dollar impact, not churn probability alone. Several patients with moderate churn rate (e.g. 0.39, 0.31) rank in the top 10 due to high monthly revenue, illustrating why revenue-weighted outreach differs from a probability-only approach.

## Limitations
- dataset is fully synthetic, so findings reflect designed relationships rather than real-world behavior
- MDI feature importance initially overstated two random noise features and understated a real categorical driver (**Insurance_Type**), corrected via permutation importance. (This is worth noting as a general modeling lesson, not just a project quirk)
- no cost data for retention outreach, so no ROI calculation is included
- model would need retraining and revalidation against real patient data before any production use

---

## Setup

**Prerequisites:**
- Python 3.9+
- pip

**Installation:**
1. Clone the repo:
   ```bash
   git clone https://github.com/jlynne2004/patient-churn-rate.git
   cd patient-churn-rate
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**File Structure:**
```
patient-churn-rate/
├── raw/ # Generated dataset (added to .gitignore, reproducable via seed)
├── .gitignore
├── LICENSE
├── patient_churn_analysis.ipynb # EDA, model training, evaluation, revenue-at-risk analysis
├── README.md
├── synthetic_patient_dataset.py # Generates the synthetic patient dataset (CSV)
```

**How to run:**
1. Run the generator to produce the dataset:
   ```bash
   python synthetic_patient_dataset.py
   ```
   This outputs `patient_churn_dataset.csv` to the project directory.
2. Launch Jupyter and open the analysis notebook:
   ```bash
   jupyter notebook patient_churn_analysis.ipynb
   ```
3. Run all cells top to bottom.

