import pandas as pd

excel_path = "data/Iotox Data from Suparno Chowdhury.xlsx"
sheet_name = "Sheet1"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

columns_to_drop = []

for col in df.columns:
    if col.endswith(".1"):
        base_col = col[:-2]  # remove ".1"

        if base_col in df.columns:
            # Compare after filling NaN
            col1 = df[base_col].fillna("").astype(str).str.strip()
            col2 = df[col].fillna("").astype(str).str.strip()

            if col1.equals(col2):
                print(f"Dropping duplicate column: {col}")
                columns_to_drop.append(col)
            else:
                print(f"Keeping {col} (not identical to {base_col})")

# Drop identical duplicate columns
df = df.drop(columns=columns_to_drop)

# print("\nFinal columns:")
# print(df.columns)

# Clean column names
df.columns = df.columns.str.strip()

# # Apply WHERE conditions
# filtered = df[
#     (df["Cancer type"] == "Colorectal") 
#     & (df["Number of arms"] >= 4) 
#     & (df["Control arm"] == "Anti-EGFR+Radiotherapy")
#     | (df["Control arm"] == "Chemo")
# ]

filtered = df[
    (df["Cancer type"] == "Renal cell") 
    & (df["Name of ICI"] == "Nivolumab") 
    # & (df["Name of ICI"] == "Tremelimumab") 
    & (df["Monotherapy/combination"] == "Combination")
]

print("\nFiltered DataFrame:")
print(filtered)
