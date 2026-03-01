import pandas as pd
import json

# === Step 1: Load Excel file ===
df = pd.read_excel('data/Cat2 Query SQL.xlsx', sheet_name='Sheet1')

# === Step 2: Keep only required columns ===
required_columns = [
    'natural_language_query',
    'sql_query',
    'filters',
    'columns'
]

df = df[required_columns].dropna()

# === Step 3: Convert to structured JSONL ===
output_path = 'data/dataset.jsonl'

with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        record = {
            "question": row['natural_language_query'],
            "gt_sql": row['sql_query'],
            "gt_filters": row['filters'],
            "gt_columns": row['columns']
        }
        
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Structured JSONL dataset saved to {output_path}")
