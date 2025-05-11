import sqlglot
from sqlglot.errors import ParseError
from typing import Dict, Any, List
from schema_loader import SchemaLoader

REQUIRED_FIELDS = ["nl", "sql", "tables", "columns", "business_intent", "complexity"]

class Validator:
    def __init__(self, schema_loader: SchemaLoader, dialect: str = "snowflake"):
        self.schema_loader = schema_loader
        self.dialect = dialect

    def validate_sql(self, sql: str) -> bool:
        try:
            sqlglot.parse_one(sql, read=self.dialect)
            return True
        except ParseError as e:
            print(f"SQL Parse Error: {e}")
            return False

    def validate_schema_references(self, tables: List[str], columns: List[str]) -> bool:
        all_tables = set(self.schema_loader.get_all_table_names())
        for t in tables:
            if t not in all_tables:
                print(f"Table not in schema: {t}")
                return False
        for col in columns:
            if "." in col:
                t, c = col.split(".", 1)
                table = self.schema_loader.get_table(t)
                if not table or c not in table.columns:
                    print(f"Column not in schema: {col}")
                    return False
        return True

    def validate_metadata(self, example: Dict[str, Any]) -> bool:
        for field in REQUIRED_FIELDS:
            if field not in example or not example[field]:
                print(f"Missing or empty field: {field}")
                return False
        return True

    def validate_nl(self, nl: str) -> bool:
        return isinstance(nl, str) and len(nl.strip()) > 5

    def validate_example(self, example: Dict[str, Any]) -> bool:
        if not self.validate_metadata(example):
            return False
        if not self.validate_nl(example["nl"]):
            print("NL question too short or empty.")
            return False
        if not self.validate_sql(example["sql"]):
            return False
        if not self.validate_schema_references(example["tables"], example["columns"]):
            return False
        return True

if __name__ == "__main__":
    loader = SchemaLoader()
    validator = Validator(loader)
    example = {
        "nl": "How many items were sold last month?",
        "sql": "SELECT COUNT(*) FROM store_sales WHERE sale_date BETWEEN '2023-03-01' AND '2023-03-31'",
        "tables": ["store_sales"],
        "columns": ["store_sales.sale_date", "store_sales.count(*)"],
        "business_intent": "Sales analysis",
        "complexity": "easy"
    }
    print("Valid example:", validator.validate_example(example))
    bad_example = {
        "nl": "How many?",
        "sql": "SELECT * FROM not_a_table",
        "tables": ["not_a_table"],
        "columns": ["not_a_table.id"],
        "business_intent": "Test",
        "complexity": "easy"
    }
    print("Invalid example:", validator.validate_example(bad_example)) 