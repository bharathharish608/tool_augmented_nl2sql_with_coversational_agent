import sqlglot
from typing import Dict, Any, List

# Helper to filter only real schema tables/columns
def filter_schema_tables_columns(tables, columns, schema_loader):
    schema_tables = set(schema_loader.get_all_table_names())
    filtered_tables = [t for t in tables if t in schema_tables]
    filtered_columns = []
    for col in columns:
        if "." in col:
            t, c = col.split(".", 1)
            if t in schema_tables:
                table = schema_loader.get_table(t)
                if table and c in table.columns:
                    filtered_columns.append(col)
    return filtered_tables, filtered_columns

def extract_tables_and_columns(sql: str, dialect: str = "snowflake"):
    parsed = sqlglot.parse_one(sql, read=dialect)
    tables = set()
    columns = set()
    for table in parsed.find_all(sqlglot.exp.Table):
        tables.add(table.name)
    for col in parsed.find_all(sqlglot.exp.Column):
        if col.table:
            columns.add(f"{col.table}.{col.name}")
        else:
            # If only one table, qualify the column
            if len(tables) == 1:
                table = next(iter(tables))
                columns.add(f"{table}.{col.name}")
            else:
                columns.add(col.name)
    return list(tables), list(columns)

def infer_sql_features(sql: str, dialect: str = "snowflake") -> List[str]:
    parsed = sqlglot.parse_one(sql, read=dialect)
    features = []
    if any(isinstance(node, sqlglot.exp.Join) for node in parsed.walk()):
        features.append("join")
    if any(isinstance(node, sqlglot.exp.Func) and node.name and node.name.upper() in {"COUNT", "SUM", "AVG", "MIN", "MAX"} for node in parsed.walk()):
        features.append("aggregation")
    if parsed.args.get("where"):
        features.append("filter")
    if parsed.args.get("group"):
        features.append("group_by")
    if parsed.args.get("order"):
        features.append("order_by")
    if any(isinstance(node, sqlglot.exp.Subquery) for node in parsed.walk()):
        features.append("subquery")
    return features

def infer_complexity(features: List[str]) -> str:
    if "join" in features or "subquery" in features:
        return "hard"
    if "aggregation" in features or "group_by" in features:
        return "medium"
    return "easy"

class MetadataAnnotator:
    def __init__(self, schema_loader, dialect: str = "snowflake"):
        self.schema_loader = schema_loader
        self.dialect = dialect

    def annotate(self, example: Dict[str, Any]) -> Dict[str, Any]:
        sql = example["sql"]
        tables, columns = extract_tables_and_columns(sql, self.dialect)
        # Filter only real schema tables/columns
        tables, columns = filter_schema_tables_columns(tables, columns, self.schema_loader)
        features = infer_sql_features(sql, self.dialect)
        example["tables"] = tables
        example["columns"] = columns
        example["sql_features"] = features
        if "complexity" not in example or not example["complexity"]:
            example["complexity"] = infer_complexity(features)
        # Ensure paraphrases is always a list
        if "paraphrases" in example:
            if isinstance(example["paraphrases"], str):
                # Try to split by common delimiters
                if ";" in example["paraphrases"]:
                    example["paraphrases"] = [p.strip() for p in example["paraphrases"].split(";") if p.strip()]
                elif "," in example["paraphrases"]:
                    example["paraphrases"] = [p.strip() for p in example["paraphrases"].split(",") if p.strip()]
                else:
                    example["paraphrases"] = [example["paraphrases"]]
            elif not isinstance(example["paraphrases"], list):
                example["paraphrases"] = [str(example["paraphrases"])]
        return example

if __name__ == "__main__":
    from schema_loader import SchemaLoader
    loader = SchemaLoader()
    annotator = MetadataAnnotator(loader)
    example = {
        "nl": "How many items were sold last month?",
        "sql": "SELECT COUNT(*) as total_sales FROM store_sales WHERE sale_date BETWEEN '2023-03-01' AND '2023-03-31'",
        "paraphrases": "How many products were sold in March?;Total items sold last month?"
    }
    annotated = annotator.annotate(example)
    print("Annotated example:", annotated) 