import pandas as pd
import sqlite3
import json
import re

# ===============================
# STEP 1: Load & STRONGLY Clean Excel
# ===============================

excel_path = "data/Iotox Data from Suparno Chowdhury.xlsx"
sheet_name = "Sheet1"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# ---- Clean column names ----
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print("Original Columns:")
print(list(df.columns))


# ===============================
# STEP 1.5: Remove True Duplicate Columns (.1, .2 etc.)
# ===============================

columns_to_drop = []
checked_pairs = []

for col in df.columns:
    # Match columns like "Column.1", "Column.2"
    match = re.match(r"^(.*)\.(\d+)$", col)
    
    if match:
        base_col = match.group(1)

        # If original base column exists
        if base_col in df.columns:
            checked_pairs.append((base_col, col))

            # Compare full column values (NaN-safe comparison)
            if df[base_col].equals(df[col]):
                columns_to_drop.append(col)
                print(f"Dropping identical duplicate column: {col}")
            else:
                print(f"Column name duplicate but values differ: {base_col} vs {col}")

# Drop only fully identical columns
if columns_to_drop:
    df = df.drop(columns=columns_to_drop)

print("\nFinal Columns After Deduplication:")
print(list(df.columns))

with open("data/schema.json", "w") as f:
    json.dump(list(df.columns), f, indent=4)


# ===============================
# STEP 2: Create SQLite Database
# ===============================

conn = sqlite3.connect("data/database.db")
table_name = "clinical_trials"

df.to_sql(table_name, conn, if_exists="replace", index=False)

print(f"\nDatabase created with normalized data: {table_name}")


# ===============================
# STEP 3: Execute SQL Queries Safely
# ===============================

input_jsonl = "data/dataset.jsonl"
output_jsonl = "data/dataset_with_answers.jsonl"

with open(input_jsonl, "r", encoding="utf-8") as infile, \
     open(output_jsonl, "w", encoding="utf-8") as outfile:

    for line_number, line in enumerate(infile, start=1):

        record = json.loads(line.strip())
        sql_query = record.get("gt_sql", "").strip()

        if not sql_query:
            record["answer"] = "SQL_ERROR: Empty SQL query"
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        try:
            cursor = conn.execute(sql_query)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                answer = [dict(zip(columns, row)) for row in rows]
            else:
                answer = []

        except Exception as e:
            answer = f"SQL_ERROR: {str(e)}"
            print(f"[ERROR] Line {line_number}: {e}")

        record["answer"] = answer
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nDataset updated with normalized answers successfully.")

conn.close()
