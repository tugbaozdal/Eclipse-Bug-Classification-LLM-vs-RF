import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

# 1. Configuration and Paths
# We assume Eclipse.csv is in the root directory
INPUT_FILE = 'data/Eclipse.csv'
OUTPUT_TRAIN = 'data/train_data.csv'
OUTPUT_TEST = 'data/test_data.csv'

def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation and numbers."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def preprocess_data():
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # 2. Load Raw Data
    print(f"📂 Loading raw data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found!")
        return

    df = pd.read_csv(INPUT_FILE, sep=';')

    # 3. Filter Target Components
    target_components = ['Core', 'UI', 'Releng', 'Diagram', 'SWT']
    df_filtered = df[df['co'].isin(target_components)].copy()

    # 4. Balancing Classes
    # Taking 200 samples from each class to ensure a balanced dataset (1000 total)
    print("⚖️ Balancing classes (200 samples per unit)...")
    balanced_df = df_filtered.groupby('co').sample(n=200, random_state=42)

    # 5. Text Cleaning
    print("🧹 Cleaning text data...")
    balanced_df['clean_text'] = balanced_df['sd'].apply(clean_text)

    # 6. Stratified Train-Test Split (80% Train, 20% Test)
    # Using 'stratify' to maintain equal class distribution in both sets
    print("✂️ Splitting data into train and test sets...")
    train_df, test_df = train_test_split(
        balanced_df, 
        test_size=0.20, 
        stratify=balanced_df['co'], 
        random_state=42
    )

    # 7. Select Necessary Columns and Save
    cols = ['bugID', 'sd', 'clean_text', 'co']
    train_df = train_df[cols]
    test_df = test_df[cols]

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print("\n" + "="*40)
    print("📊 DATA PREPROCESSING COMPLETE")
    print("="*40)
    print(f"Total Training Samples: {len(train_df)}")
    print(f"Total Testing Samples:  {len(test_df)}")
    print("\nClass distribution in test set:\n", test_df['co'].value_counts())
    print(f"\n✅ Files saved to 'data/' folder.")

if __name__ == "__main__":
    preprocess_data()