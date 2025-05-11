import csv
import json
import os
from typing import List, Dict, Any, Set, Tuple

class DatasetWriter:
    def __init__(self):
        pass

    @staticmethod
    def deduplicate(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, str]] = set()
        deduped = []
        for ex in examples:
            key = (ex.get("nl", ""), ex.get("sql", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(ex)
        return deduped

    @staticmethod
    def write_csv(examples: List[Dict[str, Any]], path: str, fieldnames: List[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ex in examples:
                row = {}
                for field in fieldnames:
                    val = ex.get(field, "")
                    if isinstance(val, list):
                        row[field] = ";".join(str(v) for v in val)
                    elif isinstance(val, dict):
                        row[field] = json.dumps(val)
                    else:
                        row[field] = val
                writer.writerow(row)

    @staticmethod
    def write_jsonl(examples: List[Dict[str, Any]], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Minimal usage example
    examples = [
        {
            "nl": "How many items were sold last month?",
            "paraphrases": ["What is the total sales for last month?", "How much did we sell last month?"],
            "sql": "SELECT COUNT(*) FROM store_sales WHERE ...",
            "tables": ["store_sales"],
            "columns": ["store_sales.count(*)"],
            "business_intent": "Sales analysis",
            "complexity": "easy",
            "alternatives": ["How many products were sold in March?"],
            "label": 1
        },
        {
            "nl": "How many items were sold last month?",
            "paraphrases": ["What is the total sales for last month?", "How much did we sell last month?"],
            "sql": "SELECT COUNT(*) FROM store_sales WHERE ...",
            "tables": ["store_sales"],
            "columns": ["store_sales.count(*)"],
            "business_intent": "Sales analysis",
            "complexity": "easy",
            "alternatives": ["How many products were sold in March?"],
            "label": 1
        },
        {
            "nl": "List all customers from California.",
            "paraphrases": ["Show me all customers in CA", "Give me a list of customers from California"],
            "sql": "SELECT * FROM customers WHERE state = 'CA'",
            "tables": ["customers"],
            "columns": ["customers.state", "customers.*"],
            "business_intent": "Customer segmentation",
            "complexity": "easy",
            "alternatives": ["List customers by state"],
            "label": 1
        }
    ]
    deduped = DatasetWriter.deduplicate(examples)
    fieldnames = ["nl", "paraphrases", "sql", "tables", "columns", "business_intent", "complexity", "alternatives", "label"]
    DatasetWriter.write_csv(deduped, "labelled_dataset_gen/datasets/sample_dataset.csv", fieldnames)
    DatasetWriter.write_jsonl(deduped, "labelled_dataset_gen/datasets/sample_dataset.jsonl")
    print("Wrote sample_dataset.csv and sample_dataset.jsonl with deduplication to labelled_dataset_gen/datasets/.") 