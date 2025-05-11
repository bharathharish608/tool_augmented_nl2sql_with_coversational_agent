import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import anthropic

# Load config from Global folder
load_dotenv(Path(__file__).parent.parent / "global" / "config.env")

class ClaudeDecomposer:
    def __init__(self, model="claude-3-7-sonnet-20250219", max_tokens=1024):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def build_prompt(self, nl_query: str) -> str:
        prompt = """
You are an expert SQL query architect specializing in translating complex natural language analytical questions into structured, machine-readable intermediate representations (IRs) in JSON format.

Your task is to:
1. Understand the analytical goal of the question.
2. Decompose it into logical steps that reflect how SQL would be constructed.
3. Identify all required entities (tables), dimensions, filters, metrics, groupings, join conditions, and implicit constraints.
4. Generate a structured IR (in JSON) containing:
   - Query understanding
   - Entities (primary and secondary, with join keys embedded as join_condition if mentioned)
   - Metrics and measures
   - Filter conditions
   - Time dimensions
   - Step-by-step SQL execution plan
   - Final single-line SQL pseudocode (complete_sql_pseudocode) using CTEs with no newline characters
5. Think step by step — especially for multi-hop reasoning, subqueries, nested filters, ranking, or grouping logic.

---

Here is the natural language analytical question:
"""
        prompt += f'"{nl_query}"\n'
        prompt += "Return only the IR in JSON format with no explanations."
        return prompt

    def decompose(self, nl_query: str) -> Dict[str, Any]:
        prompt = self.build_prompt(nl_query)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content
            # Handle list response from Claude (TextBlock or similar objects)
            if isinstance(response, list):
                try:
                    response = " ".join(getattr(part, 'text', str(part)) for part in response)
                except Exception:
                    response = str(response)
            print("\n--- RAW CLAUDE RESPONSE ---\n")
            print(response)
            print("\n--- END RAW RESPONSE ---\n")
            # Extract JSON from response
            try:
                if isinstance(response, str) and response.strip().startswith("```json"):
                    response = response.strip().split("```json", 1)[1].split("```", 1)[0].strip()
                return json.loads(response)
            except Exception:
                import re
                if isinstance(response, str):
                    match = re.search(r'\{.*\}', response, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
                raise ValueError("Could not extract JSON from Claude response")
        except Exception as e:
            print(f"Claude API call failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

if __name__ == "__main__":
    # Example usage
    nl_query = "Find the top 3 products (by total sales amount) sold in the last 90 days, where the product has at least 5 unique buyers and the average order value is above $100. Also include the product's category and average rating, and return the results sorted by sales amount in descending order."
    decomposer = ClaudeDecomposer()
    ir = decomposer.decompose(nl_query)
    print(json.dumps(ir, indent=2)) 