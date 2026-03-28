import pandas as pd
from google import genai
import time
import os
from tqdm import tqdm
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# 1. Configuration and API Setup
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file!")

client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-flash-lite"

# File Paths
input_file = 'data/test_data.csv'
output_file = 'data/gemini_final_results.csv'

# 2. Resuming Logic (Checkpointing)
df_test = pd.read_csv(input_file)

if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    processed_ids = df_existing['bugID'].tolist()
    df_to_process = df_test[~df_test['bugID'].isin(processed_ids)].copy()
    print(f"🔄 {len(processed_ids)} records already processed. Resuming with {len(df_to_process)} records...")
else:
    df_to_process = df_test.copy()
    print("🚀 Classification marathon starting from scratch...")

# 3. Robust Prediction Function
def get_prediction(summary, max_retries=5):
    prompt = f"""You are an Eclipse software expert. Review the following bug report summary 
    and classify it into one of these 5 technical units: Core, UI, Releng, Diagram, SWT.
    RULE: Return ONLY the unit name. No extra explanation.
    Bug Report Summary: {summary}
    Unit:"""
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            return response.text.strip().replace("`", "")
        except Exception as e:
            wait = (2 ** attempt)
            time.sleep(wait)
    return "Error"

# 4. Processing Loop
for i in tqdm(range(len(df_to_process)), desc="Classifying"):
    row = df_to_process.iloc[i]
    prediction = get_prediction(row['sd'])
    
    new_row = pd.DataFrame([[row['bugID'], row['co'], prediction]], 
                              columns=['bugID', 'actual_co', 'gemini_prediction'])
    
    new_row.to_csv(output_file, mode='a', index=False, 
                      header=not os.path.exists(output_file), encoding="utf-8")
    
    time.sleep(0.5)

# 5. Final Report
print("\n🎯 Processing complete! Analyzing results...")
final_df = pd.read_csv(output_file)
valid_labels = ['Core', 'UI', 'Releng', 'Diagram', 'SWT']

final_df['gemini_prediction_cleaned'] = final_df['gemini_prediction'].apply(
    lambda x: x if x in valid_labels else 'Other'
)

print("\n" + "="*40)
print("GEMINI 2.5 FLASH-LITE PERFORMANCE REPORT")
print("="*40)
print(classification_report(final_df['actual_co'], final_df['gemini_prediction_cleaned']))