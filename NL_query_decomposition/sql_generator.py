import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import json

global_config = Path(__file__).parent.parent / "global" / "config.env"
load_dotenv(global_config)

SYSTEM_PROMPT = """
You are an expert SQL query generator. Given an intermediate representation (IR) of a query, generate a valid SQL query in the specified dialect. Use CTEs if the execution plan has multiple steps. Output only the SQL query, with no extra text.
"""

def prune_ir(ir):
    """Return a concise IR with only essential fields for SQL generation."""
    if not isinstance(ir, dict):
        return ir
    keep_keys = {"entities", "metrics", "filters", "groupings", "group_by", "execution_plan"}
    return {k: v for k, v in ir.items() if k in keep_keys and v}

def generate_sql_from_ir(ir, dialect='trino', sql_pseudocode=None, nl_query=None, model="claude-3-7-sonnet-20250219", max_tokens=8192):
    """
    Generate SQL from the given IR (and optional SQL pseudocode and NL query) using Claude (Anthropic API).
    Args:
        ir (dict): The intermediate representation (IR) as a dict.
        dialect (str): The SQL dialect to generate (e.g., 'trino', 'snowflake').
        sql_pseudocode (str, optional): LLM-generated SQL pseudocode as a reference.
        nl_query (str, optional): The original natural language query.
        model (str): Claude model name.
        max_tokens (int): Max tokens for the LLM response.
    Returns:
        str: The generated SQL string.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    concise_ir = prune_ir(ir)
    prompt = """
You are an expert SQL query generator. Given the following information, generate a valid SQL query in the {dialect} dialect. Use CTEs if the execution plan has multiple steps. Output only the SQL query, with no extra text.
""".format(dialect=dialect)
    if nl_query:
        prompt += f"\nUser Query: {nl_query}"
    if concise_ir:
        prompt += f"\nConcise IR:\n{json.dumps(concise_ir, indent=2)}"
    if sql_pseudocode:
        prompt += f"\nSQL Pseudocode:\n{sql_pseudocode}"
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content
        if isinstance(response, list):
            sql = " ".join(getattr(part, 'text', str(part)) for part in response)
        else:
            sql = str(response)
        sql = sql.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()
    except Exception as e:
        return f"-- ERROR: Failed to generate SQL from IR: {e}" 