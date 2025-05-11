"""
Shared helpers for semantic schema search (data loading, normalization, etc.).
"""

import json
from pathlib import Path
import csv
import re
import numpy as np

def load_schema(schema_path):
    """
    Load schema from tpcds_with_all_descriptions.json and return a list of schema element texts.
    Each element is a string combining table name, column name, and description (if available).
    """
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    elements = []
    for table_name, table in schema["tables"].items():
        table_desc = table.get('description', '')
        elements.append(f"table: {table_name} - {table_desc}")
        columns = table.get('columns', {})
        if isinstance(columns, dict):
            for col, desc in columns.items():
                col_desc = desc.get('description') if isinstance(desc, dict) else desc
                elements.append(f"column: {col} (table: {table_name}) - {col_desc}")
        elif isinstance(columns, list):
            for col in columns:
                elements.append(f"column: {col} (table: {table_name})")
    return elements

def load_labelled_pairs(labelled_csv_path, schema_corpus=None):
    """
    Load labelled (query, relevant/irrelevant schema element) pairs from labelled_dataset.csv.
    Expands each row to use both NL and all paraphrases as queries, and pairs with all tables and columns.
    If schema_corpus is provided, matches schema elements to the exact string in the corpus (with description).
    Returns a list of (query, schema_element, label) tuples.
    """
    pairs = []
    # Build lookup for full schema_corpus strings
    table_lookup = {}
    column_lookup = {}
    if schema_corpus:
        for elem in schema_corpus:
            if elem.startswith("table: "):
                # table: table_name - description
                table_name = elem.split(": ", 1)[1].split(" - ", 1)[0].strip()
                table_lookup[table_name] = elem
            elif elem.startswith("column: "):
                # column: col (table: table_name) - description
                import re
                m = re.match(r"column: (.*?) \(table: (.*?)\)", elem)
                if m:
                    col, table = m.group(1).strip(), m.group(2).strip()
                    column_lookup[(col, table)] = elem
    import csv
    import re
    with open(labelled_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row.get('label', 1))
            # Get queries: nl + paraphrases
            queries = [row['nl'].strip()] if row['nl'].strip() else []
            # Extract paraphrases (if present)
            if row.get('paraphrases'):
                lines = row['paraphrases'].split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or re.match(r'^(#|\d+\.|Paraphrases)', line):
                        continue
                    queries.append(line)
            # Get schema elements: tables and columns
            tables = [t.strip() for t in row.get('tables', '').split(';') if t.strip()]
            columns = [c.strip() for c in row.get('columns', '').split(';') if c.strip()]
            # Pair each query with each table and column, using full schema_corpus string if available
            for query in queries:
                for table in tables:
                    schema_elem = table_lookup.get(table, f"table: {table}") if schema_corpus else f"table: {table}"
                    pairs.append((query, schema_elem, label))
                for column in columns:
                    table_context = tables[0] if tables else ''
                    schema_elem = None
                    if schema_corpus and table_context:
                        schema_elem = column_lookup.get((column, table_context))
                    if not schema_elem:
                        schema_elem = f"column: {column} (table: {table_context})" if table_context else f"column: {column}"
                    pairs.append((query, schema_elem, label))
    return pairs

def normalize_scores(scores):
    """
    Normalize a list or numpy array of scores using min-max normalization.
    If all values are the same, return zeros.
    """
    scores = np.array(scores)
    if scores.size == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val) 