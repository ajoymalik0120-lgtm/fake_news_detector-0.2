import pandas as pd
import os

input_file = 'IFND.csv' 
output_dir = 'data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f" Reading {input_file} using safe fallbacks...")
try:
    df = pd.read_csv(input_file, encoding='latin-1', engine='python')
except FileNotFoundError:
    print(f" Error: Could not find '{input_file}' in your main project folder!")
    exit()

print(" Detected Columns:", list(df.columns))

text_col = 'Statement' if 'Statement' in df.columns else ('text' if 'text' in df.columns else df.columns[0])
label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

#  FIXED: Convert to string, lowercase, and completely strip hidden spaces
df[label_col] = df[label_col].astype(str).str.lower().str.strip()

# Filter using the exact text representations found in the file
true_df = df[df[label_col].isin(['true', '1', '1.0'])][[text_col]].rename(columns={text_col: 'title'}).dropna()
fake_df = df[df[label_col].isin(['fake', 'false', '0', '0.0'])][[text_col]].rename(columns={text_col: 'title'}).dropna()

print(f" Found: {len(true_df)} True records and {len(fake_df)} Fake records.")

true_path = os.path.join(output_dir, 'true.csv')
fake_path = os.path.join(output_dir, 'fake.csv')

true_df.to_csv(true_path, index=False, encoding='utf-8')
fake_df.to_csv(fake_path, index=False, encoding='utf-8')

print(f" Success! Overwritten '{true_path}' and '{fake_path}' cleanly with extracted records.")
print(" You are now ready to run 'python train_model.py'.")
