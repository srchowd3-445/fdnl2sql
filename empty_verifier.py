import json
import sqlite3
from pathlib import Path

# ===============================
# CONFIG
# ===============================
db_path = "data/database.db"

# Your file (Windows path). Use raw string to avoid backslash escapes:
replaced_json_path = r"C:\Users\dhrub\OneDrive\Desktop\MAYO_AIM2\data\empty_gt_replaced.json"

output_json_path = "data/empty_gt_replaced_with_results.json"

# If True, include the original empty_gt_sql execution status too (should be empty, but useful to confirm)
CHECK_EMPTY_GT_TOO = False

# ===============================
# Helpers
# ===============================
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()

def execute_select_as_dicts(conn: sqlite3.Connection, sql: str):
    """
    Executes a SELECT and returns list[dict] rows.
    If it's not a SELECT or no columns, returns [].
    Raises on SQL error.
    """
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)

    if cur.description is None:
        return []

    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

# ===============================
# MAIN
# ===============================
conn = sqlite3.connect(db_path)

with open(replaced_json_path, "r", encoding="utf-8") as f:
    repaired_items = json.load(f)

out = []
total = 0
executed = 0
errors = 0
empty_new = 0

for item in repaired_items:
    total += 1

    new_sql = item.get("new_gt_sql")
    if not new_sql:
        # Nothing to execute
        out.append({
            **item,
            "new_gt_result": None,
            "new_gt_status": "NO_NEW_SQL"
        })
        continue

    try:
        result = execute_select_as_dicts(conn, new_sql)
        executed += 1

        status = "OK"
        if isinstance(result, list) and len(result) == 0:
            status = "EMPTY_NEW_SQL_RESULT"
            empty_new += 1

        out_item = {**item, "new_gt_result": result, "new_gt_status": status}

        if CHECK_EMPTY_GT_TOO:
            try:
                empty_res = execute_select_as_dicts(conn, item.get("empty_gt_sql"))
                out_item["empty_gt_result_check"] = empty_res
                out_item["empty_gt_status_check"] = "OK" if empty_res == [] else "NOT_EMPTY_UNEXPECTED"
            except Exception as e:
                out_item["empty_gt_result_check"] = None
                out_item["empty_gt_status_check"] = f"SQL_ERROR: {str(e)}"

        out.append(out_item)

    except Exception as e:
        errors += 1
        out.append({
            **item,
            "new_gt_result": None,
            "new_gt_status": f"SQL_ERROR: {str(e)}"
        })

conn.close()

# Save results
Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("===== EXECUTE REPLACED SQL REPORT =====")
print(f"Total items in replaced file: {total}")
print(f"Executed new_gt_sql: {executed}")
print(f"New SQL returned empty []: {empty_new}")
print(f"SQL errors: {errors}")
print(f"Saved output to: {output_json_path}")

# Optional: print the first few results to the console
print("\n--- Sample (first 2) ---")
for sample in out[:2]:
    print(f"\nLine {sample.get('line_number')}: {sample.get('question')}")
    print("new_gt_status:", sample.get("new_gt_status"))
    res = sample.get("new_gt_result")
    if isinstance(res, list):
        print("rows:", len(res))
        if len(res) > 0:
            print("first row:", res[0])