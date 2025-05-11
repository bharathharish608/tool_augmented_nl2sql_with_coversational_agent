import anthropic
import os
import json
from dotenv import load_dotenv

# Load API key from .env or environment
load_dotenv("global/config.env")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define the IRModel schema for function calling
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

# Define the tool for IR generation
ir_tool = {
    "name": "generate_ir",
    "description": "Generate an intermediate representation (IR) for a natural language analytical query. Output only the IR in JSON.",
    "input_schema": dir_schema
}

# Example NL query
test_query = "Find all regions where female users have the highest percentage of online orders, in descending order."

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=2048,
    tools=[ir_tool],
    messages=[
        {"role": "user", "content": test_query}
    ]
)

# Print the raw response for inspection
print(json.dumps(response.model_dump(), indent=2))

# Extract and print the IR if present
for content in response.content:
    if getattr(content, "type", None) == "tool_use" and getattr(content, "name", None) == "generate_ir":
        print("\n--- IR Output ---")
        print(json.dumps(content.input, indent=2)) 