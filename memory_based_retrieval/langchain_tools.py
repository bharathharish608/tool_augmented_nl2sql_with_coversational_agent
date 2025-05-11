from langchain.tools import Tool
from relation_aware_retrieval.relation_aware_tool import RelationAwareRetriever
from NL_query_decomposition.schema_tools import SchemaTools
from NL_query_decomposition.tool_augmented_agent import ToolAugmentedAgent
from NL_query_decomposition.sql_generator import generate_sql_from_ir
from pathlib import Path
import json
import re
import logging
from memory_based_retrieval.ir_schema import IRModel
import sqlglot
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import anthropic
from dotenv import load_dotenv
import os

def extract_sql(text):
    """
    Extract only the SQL code from LLM output, removing markdown/code blocks and explanations.
    - Handles ```sql ... ``` and generic code blocks.
    - Extracts from SELECT to last semicolon or end if possible.
    - Strips any explanations or extra text.
    """
    if not isinstance(text, str):
        return text
    # Remove all markdown code blocks
    code_block = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    code_block = re.search(r"```(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    # Extract SQL from SELECT to last semicolon (or end)
    sql_match = re.search(r"(SELECT[\s\S]+?;)", text, re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    # If no semicolon, try to extract from SELECT to end
    sql_match = re.search(r"(SELECT[\s\S]+)", text, re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    # Fallback: return the text as is, stripped
    return text.strip()

# Initialize schema tools and agent core
schema_tools = SchemaTools()
agent_core = ToolAugmentedAgent()

# List tables tool
list_tables_tool = Tool(
    name="ListTables",
    func=lambda _: schema_tools.list_tables(),
    description="Lists all tables in the schema."
)

# Get columns tool with error handling
def get_columns_tool_func(table):
    try:
        return schema_tools.get_columns(table)
    except ValueError as e:
        return f"ERROR: {e}"

get_columns_tool = Tool(
    name="GetColumns",
    func=get_columns_tool_func,
    description="Gets columns for a given table. Input: table name. Only use valid table names as found in list_tables(). If the table does not exist, you will receive an error."
)

# Search schema tool
search_schema_tool = Tool(
    name="SearchSchema",
    func=lambda keyword: schema_tools.search_schema(keyword),
    description="Searches tables and columns matching the keyword."
)

# Get column description tool
get_column_description_tool = Tool(
    name="GetColumnDescription",
    func=lambda args: schema_tools.get_column_description(*[a.strip() for a in args.split(',')]),
    description="Gets the description for a specific column. Input: 'table,column'."
)

# Semantic search tool
semantic_search_tool = Tool(
    name="SemanticSearchSchema",
    func=lambda keyword: schema_tools.semantic_search_schema(keyword),
    description="Returns top semantically matching schema elements for a keyword."
)

# Relation-aware retrieval tool (already present)
relation_aware_tool = Tool(
    name="RelationAwareRetrieval",
    func=lambda nl_query: RelationAwareRetriever(
        Path("global/tpcds_with_all_descriptions.json"),
        Path("labelled_dataset_gen/datasets/labelled_dataset.csv")
    ).retrieve(nl_query, top_k=10),
    description="Returns top-k relevant schema elements using graph-based, context-aware retrieval (RWR). Use this tool when the query is ambiguous, involves multiple tables, or when other tools do not provide sufficient context."
)

# Claude function-calling IR generation
load_dotenv("global/config.env")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

dir_schema = {
    "type": "object",
    "properties": {
        "query_understanding": {"type": "string"},
        "entities": {"type": "object"},
        "metrics": {"type": "array", "items": {}},
        "filters": {"type": "array", "items": {}},
        "time_dimensions": {"type": "array", "items": {}},
        "groupings": {"type": "array", "items": {}},
        "execution_plan": {"type": "array", "items": {"type": "string"}},
        "complete_sql_pseudocode": {"type": "string"}
    },
    "required": ["query_understanding", "entities"]
}

ir_tool = {
    "name": "generate_ir",
    "description": "Generate an intermediate representation (IR) for a natural language analytical query. Output only the IR in JSON.",
    "input_schema": dir_schema
}

def ir_generation_func(nl_query):
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        tools=[ir_tool],
        messages=[{"role": "user", "content": nl_query}]
    )
    for content in response.content:
        if getattr(content, "type", None) == "tool_use" and getattr(content, "name", None) == "generate_ir":
            return content.input
    raise ValueError("Claude did not return a valid IR tool_use block.")

ir_generation_tool = Tool(
    name="GenerateIR",
    func=ir_generation_func,
    description="Generate the IR (intermediate representation) for the given NL query using Claude 3.7 function calling. Output is always structured JSON."
)

# SQL pseudocode generation tool
def sql_pseudocode_func(ir):
    return extract_sql(agent_core.generate_sql_pseudocode(ir))

sql_pseudocode_tool = Tool(
    name="GenerateSQLPseudocode",
    func=sql_pseudocode_func,
    description="Generate SQL pseudocode for the given IR. Output ONLY the SQL code, with NO explanations, markdown, or extra text. Do NOT use code blocks. If you include anything else, the agent will ask you to try again."
)

def robust_generate_sql(args):
    # Log the input type and value for debugging
    logging.info(f"[GenerateSQL] Received args of type {type(args)}: {args}")
    # Default dialect
    dialect = 'trino'
    # Parse args and extract dialect if present
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                args = parsed
            else:
                args = {"ir": parsed}
        except Exception:
            args = {"ir": args}
    # Unwrap IR if nested under 'ir' key
    if isinstance(args, dict) and 'ir' in args:
        ir_data = args['ir']
        if 'dialect' in args:
            dialect = args['dialect']
    else:
        ir_data = args
        if isinstance(args, dict) and 'dialect' in args:
            dialect = args['dialect']
    # Validate IR with Pydantic if possible
    try:
        ir_model = IRModel.parse_obj(ir_data)
    except Exception as e:
        raise ValueError(f"IR validation failed: {e}")
    # Generate SQL using the specified dialect, but do not parse, extract, or validate
    sql_code = generate_sql_from_ir(ir_model.dict(), dialect=dialect)
    return sql_code  # Return the raw LLM output as-is

# Final SQL generation tool
sql_generation_tool = Tool(
    name="GenerateSQL",
    func=robust_generate_sql,
    description="Generate the final SQL for the given IR and pseudocode, with dialect-specific support (default: trino). Returns the raw SQL output from the LLM, including any explanations, markdown, or formatting. No parsing, extraction, or validation is performed."
)

# Collect all tools
all_tools = [
    list_tables_tool,
    get_columns_tool,
    search_schema_tool,
    get_column_description_tool,
    semantic_search_tool,
    relation_aware_tool,
    ir_generation_tool,
    sql_pseudocode_tool,
    sql_generation_tool
] 