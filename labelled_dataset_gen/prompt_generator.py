import random
from typing import List, Dict, Optional, Any
from schema_loader import SchemaLoader, Table, Column

PROMPT_TEMPLATE = """
You are an expert data analyst. Given the following database schema context, generate:
- A natural language (NL) question
- 2-3 NL paraphrases
- The corresponding SQL query
- The tables and columns used
- The business intent
- The query complexity (easy/medium/hard)
- (Optional) Alternatives (other ways to ask or solve)

**Use only the provided schema.**
**Output your answer in the following strict JSON format:**

{{
  "nl": "...",
  "paraphrases": ["...", "..."],
  "sql": "...",
  "tables": ["..."],
  "columns": ["..."],
  "business_intent": "...",
  "complexity": "...",
  "alternatives": ["..."]
}}

Schema context:
{schema_context}

{example_section}
"""

def format_schema_context(tables: List[Table], max_columns: int = 5) -> str:
    lines = []
    for table in tables:
        lines.append(f"Table: {table.name}")
        lines.append(f"  Description: {table.description}")
        col_names = list(table.columns.keys())[:max_columns]
        for col in col_names:
            col_obj = table.columns[col]
            lines.append(f"    - {col_obj.name} ({col_obj.data_type}): {col_obj.description}")
        if len(table.columns) > max_columns:
            lines.append(f"    ... and {len(table.columns) - max_columns} more columns.")
    return '\n'.join(lines)

def format_examples(examples: Optional[List[Dict[str, Any]]]) -> str:
    if not examples:
        return ""
    lines = ["\nExample NL/SQL pairs:"]
    for ex in examples:
        lines.append(f"- NL: {ex.get('nl')}")
        lines.append(f"  SQL: {ex.get('sql')}")
    return '\n'.join(lines)

class PromptGenerator:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def generate_prompt(
        self,
        table_names: List[str],
        business_intent: str,
        complexity: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        max_columns: int = 5
    ) -> str:
        tables = [self.schema_loader.get_table(t) for t in table_names if self.schema_loader.get_table(t)]
        schema_context = format_schema_context(tables, max_columns)
        example_section = format_examples(examples)
        return PROMPT_TEMPLATE.format(
            schema_context=schema_context,
            example_section=example_section
        )

if __name__ == "__main__":
    # Minimal usage example
    loader = SchemaLoader()
    generator = PromptGenerator(loader)
    # Pick 2 random tables
    all_tables = loader.get_all_table_names()
    selected_tables = random.sample(all_tables, 2)
    prompt = generator.generate_prompt(
        table_names=selected_tables,
        business_intent="Sales analysis",
        complexity="medium",
        examples=[{"nl": "How many items were sold last month?", "sql": "SELECT COUNT(*) FROM store_sales WHERE ..."}]
    )
    print(prompt) 