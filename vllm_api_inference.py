import argparse
import json
import os
import math
from typing import List, Dict

from pydantic import BaseModel
from openai import OpenAI
from openai.types import ResponseFormatJSONSchema
from structured_logprobs import add_logprobs


# ===============================
# OUTPUT JSON SCHEMA (Pydantic)
# ===============================

class SQLResponse(BaseModel):
    sql: str
    filters: Dict
    columns: List[str]


# ===============================
# HELPER FUNCTIONS
# ===============================

def exp_structure(x):
    """Recursively apply exp() to floats inside nested structures."""
    if isinstance(x, dict):
        return {k: exp_structure(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [exp_structure(v) for v in x]
    elif isinstance(x, (int, float)):
        return float(math.exp(x))
    else:
        return x


def flatten_numbers(x):
    """Recursively collect all numeric values from nested structure."""
    if isinstance(x, dict):
        vals = []
        for v in x.values():
            vals.extend(flatten_numbers(v))
        return vals
    elif isinstance(x, list):
        vals = []
        for v in x:
            vals.extend(flatten_numbers(v))
        return vals
    elif isinstance(x, (int, float)):
        return [x]
    else:
        return []


# ===============================
# PROMPT BUILDER
# ===============================

def build_messages(question: str, schema_string: str):
    return [{"role": "user", "content": f"""
You are an expert SQLite query generator.
Generate a valid SQLite query using only the schema below.

Schema:
{schema_string}

Question:
{question}

Return ONLY valid JSON.
"""}]


# ===============================
# MAIN
# ===============================

def main(args):

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.error_file), exist_ok=True)

    # Load DB schema columns
    with open(args.schema_file, "r", encoding="utf-8") as f:
        schema_columns = json.load(f)

    schema_string = f"Table: {args.table_name}\nColumns:\n"
    for col in schema_columns:
        schema_string += f'- "{col}"\n'

    # Convert Pydantic schema → OpenAI JSON schema
    json_schema = SQLResponse.model_json_schema()

    response_schema = ResponseFormatJSONSchema.model_validate({
        "type": "json_schema",
        "json_schema": {
            "name": "SQLResponse",
            "schema": json_schema
        }
    })

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Resume support
    processed = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)

    print(f"Resuming from line {processed + 1}")

    with open(args.input_file, "r", encoding="utf-8") as infile, \
         open(args.output_file, "a", encoding="utf-8") as outfile, \
         open(args.error_file, "a", encoding="utf-8") as errfile:

        for line_number, line in enumerate(infile, start=1):

            if line_number <= processed:
                continue

            record = {}
            question = ""

            try:
                record = json.loads(line.strip())
                question = record.get("question", "").strip()

                if not question:
                    record.update({
                        "error": "Empty question",
                        "pred_sql": "",
                        "pred_filters": {},
                        "pred_columns": []
                    })
                    outfile.write(json.dumps(record) + "\n")
                    outfile.flush()
                    continue

                # ===============================
                # Call OpenAI API
                # ===============================

                completion = client.chat.completions.create(
                    model=args.model_name,
                    messages=build_messages(question, schema_string),
                    logprobs=True,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    response_format=response_schema.model_dump(by_alias=True),
                )

                # Wrap for structured logprobs
                completion_with_logprobs = add_logprobs(completion)

                # Extract JSON content from ORIGINAL completion
                content = completion.choices[0].message.content
                parsed_json = json.loads(content)

                record["pred_sql"] = parsed_json.get("sql", "")
                record["pred_filters"] = parsed_json.get("filters", {})
                record["pred_columns"] = parsed_json.get("columns", [])
                record["raw_model_output"] = content

                # ===============================
                # Extract structured logprobs
                # ===============================

                field_logprobs = completion_with_logprobs.log_probs[0]
                record["field_logprobs"] = field_logprobs

                # Recursively compute confidence
                field_confidence = exp_structure(field_logprobs)
                record["field_confidence"] = field_confidence

                # Compute overall confidence
                flat_values = flatten_numbers(field_logprobs)

                if flat_values:
                    mean_logprob = sum(flat_values) / len(flat_values)
                    record["confidence_overall"] = float(math.exp(mean_logprob))
                else:
                    record["confidence_overall"] = None

            except Exception as e:
                record = {
                    "question": question,
                    "pred_sql": "",
                    "pred_filters": {},
                    "pred_columns": [],
                    "field_logprobs": {},
                    "field_confidence": {},
                    "confidence_overall": None,
                    "error": str(e)
                }
                errfile.write(json.dumps(record) + "\n")
                errfile.flush()

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            outfile.flush()

            print(f"[{line_number}] Saved instance.")

    print("Done.")


# ===============================
# ARGPARSE
# ===============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Structured SQL Generation with Field Confidence"
    )

    parser.add_argument("--api_base", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--error_file", type=str, required=True)
    parser.add_argument("--schema_file", type=str, default="data/schema.json")
    parser.add_argument("--table_name", type=str, default="clinical_trials")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
