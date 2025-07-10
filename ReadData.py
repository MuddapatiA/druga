import pandas as pd
from IPython.display import display
import re

# Define the file path and sheet names
excel_file_path = 'Inputdata.xlsx'
sheet_names = {
    'txn': 'fact_txn',
    'patient': 'dim_patient',
    'physician': 'dim_physician'
}

try:
    # Read each sheet from the Excel file into a separate DataFrame
    fact_txn = pd.read_excel(excel_file_path, sheet_name=sheet_names['txn'])
    dim_patient = pd.read_excel(excel_file_path, sheet_name=sheet_names['patient'])
    dim_physician = pd.read_excel(excel_file_path, sheet_name=sheet_names['physician'])

    print("✅ DataFrames loaded successfully from Excel file!")
    print("\nfact_txn preview:")
    display(fact_txn.head())

except FileNotFoundError:
    print(f"❌ Error: The file '{excel_file_path}' was not found.")
    print("Please make sure the Excel file is in the same directory as your notebook or script.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
    print("Please check that the sheet names ('fact_txn', 'dim_patient', 'dim_physician') are correct in your Excel file.")


 # --- 1. Data Cleaning ---

# NEW: Standardize all column names to lowercase
for df in [fact_txn, dim_patient, dim_physician]:
    df.columns = df.columns.str.lower()

# Standardize string data to lowercase
for df in [fact_txn, dim_patient, dim_physician]:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()

# CORRECTED: Convert transaction date to datetime, using the new lowercase column name
fact_txn['txn_dt'] = pd.to_datetime(fact_txn['txn_dt'])

print("✅ Data cleaning complete.")

# --- 2. Merging DataFrames ---
try:
    # Merge using the lowercase column names
    merged_df = pd.merge(fact_txn, dim_patient, on='patient_id', how='left')
    full_df = pd.merge(merged_df, dim_physician, on='physician_id', how='left')

    print("✅ DataFrames merged successfully.")
    print("\n--- Merged DataFrame Info ---")
    full_df.info()

except KeyError as e:
    print(f"❌ Merge failed. A key column is missing: {e}")
except Exception as e:
    print(f"❌ An error occurred during merging: {e}")


    # --- 3. Feature Engineering ---

print("Starting feature engineering...")

# Create a dictionary to hold our aggregated data
model_table = {}

# Aggregate basic features, taking the first entry for each patient
# We assume the first transaction corresponds to the primary diagnosis info
agg_funs = {
    'txn_dt': 'min', # Get the earliest transaction date as the diagnosis date
    'gender_x': 'first',
    'birth_year_x': 'first',
    'physician_type': 'first',
    'state': 'first',
    'txn_location_type': 'first'
}
model_table = full_df.groupby('patient_id').agg(agg_funs)
print("✅ Basic aggregations complete.")

# --- Calculate complex and new features ---

# TARGET: 1 if patient received 'drug a', 0 otherwise
treatments = full_df[full_df['txn_type'] == 'treatments']
treated_patients = treatments[treatments['txn_desc'].str.contains('drug a', na=False)]['patient_id'].unique()
model_table['target'] = model_table.index.isin(treated_patients).astype(int)

# NUM_CONDITIONS: Count of unique high-risk conditions per patient
conditions = full_df[full_df['txn_type'] == 'conditions']
num_conditions = conditions.groupby('patient_id')['txn_desc'].nunique()
model_table['num_conditions'] = num_conditions.reindex(model_table.index).fillna(0)

# NEW FEATURE 1: NUM_CONTRAINDICATIONS
contraindications = full_df[full_df['txn_type'] == 'contraindications']
num_contraindications = contraindications.groupby('patient_id')['txn_desc'].nunique()
model_table['num_contraindications'] = num_contraindications.reindex(model_table.index).fillna(0)

print("✅ Custom features calculated.")

# --- Finalize columns and calculate age-based features ---

# PATIENT_AGE: Calculated at the time of diagnosis
model_table['patient_age'] = model_table['txn_dt'].dt.year - model_table['birth_year_x']

# NEW FEATURE 2: PATIENT_IS_HIGH_RISK
model_table['patient_is_high_risk'] = ((model_table['patient_age'] >= 65) | (model_table['num_conditions'] > 0)).astype(int)

# Rename columns to match the final schema
model_table = model_table.rename(columns={
    'txn_dt': 'diseasex_dt',
    'gender_x': 'patient_gender',
    'physician_type': 'physician_type',
    'state': 'physician_state',
    'txn_location_type': 'location_type'
})

# Select the final columns for the model
final_columns = [
    'target',
    'diseasex_dt',
    'patient_age',
    'patient_gender',
    'num_conditions',
    'physician_type',
    'physician_state',
    'location_type',
    'num_contraindications', # New feature
    'patient_is_high_risk'   # New feature
]
model_table = model_table[final_columns]

# Fill any remaining missing values
model_table['physician_type'].fillna('unknown', inplace=True)
model_table['physician_state'].fillna('unknown', inplace=True)
model_table['location_type'].fillna('unknown', inplace=True)


print("\n--- Final `model_table` Info ---")
model_table.info()

print("\n--- Final `model_table` Preview ---")
display(model_table.head())

# --- Filter for Eligible Population ---
# Per the requirement, Drug A is for patients 12 and older.
eligible_patients_df = model_table[model_table['patient_age'] >= 12].copy()

print(f"Original patient count: {len(model_table)}")
print(f"Eligible patient count (age >= 12): {len(eligible_patients_df)}")
print("✅ Population filtered for eligibility.")


# --- 4. Model Building ---
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("\nStarting model training process...")

# Define features (X) and target (y)
X = eligible_patients_df.drop(columns=['target', 'diseasex_dt']) # Drop non-feature columns
y = eligible_patients_df['target']

# --- Preprocessing ---
# Convert categorical columns to a format the model can understand.
# We'll use one-hot encoding.
X_encoded = pd.get_dummies(X, columns=[
    'patient_gender',
    'physician_type',
    'physician_state',
    'location_type'
], drop_first=True) # drop_first helps avoid multicollinearity

# --- NEW: Clean the column names ---
# Replace special JSON characters with underscores
X_encoded.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X_encoded.columns]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data split into training and testing sets.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# Initialize and train the LightGBM model
lgbm = lgb.LGBMClassifier(objective='binary', random_state=42)
lgbm.fit(X_train, y_train)

print("✅ Model training complete.")


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Make Predictions on the Test Set ---
y_pred = lgbm.predict(X_test)
y_pred_proba = lgbm.predict_proba(X_test)[:, 1] # Probability of class 1 (Treated)

print("--- Model Performance on Test Data ---")

# --- 2. Display Classification Metrics ---
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Treated (0)', 'Treated (1)']))

# --- 3. Display ROC AUC Score ---
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {auc_score:.4f}")

# --- 4. Visualize the Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Treated', 'Predicted Treated'],
            yticklabels=['Actual Not Treated', 'Actual Treated'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# --- 5. Visualize Feature Importance ---
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm.feature_importances_
}).sort_values('importance', ascending=False).head(15) # Top 15 features

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Top 15 Most Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

import joblib
import json
import os # Import the os module

# --- Define the directory and filenames ---
model_dir = 'models'
model_filename = os.path.join(model_dir, 'drug_a_predictor.joblib')
columns_filename = os.path.join(model_dir, 'model_columns.json')

# --- Create the directory if it doesn't exist ---
os.makedirs(model_dir, exist_ok=True)

# --- Save the Trained Model ---
joblib.dump(lgbm, model_filename)

# --- Save the column list used for training ---
with open(columns_filename, 'w') as f:
    json.dump(list(X_train.columns), f)

print(f"✅ Model saved to {model_filename}")
print(f"✅ Model columns saved to {columns_filename}")