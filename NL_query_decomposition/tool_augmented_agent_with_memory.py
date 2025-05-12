"""
System Prompt Template:
You are an expert SQL query architect specializing in translating complex natural language analytical questions into structured, machine-readable intermediate representations (IRs) in JSON format.

You have access to the following tools for schema exploration:
- list_tables(): Returns all table names in the database.
- get_columns(table_name): Returns all columns (and their descriptions) for a given table.
- search_schema(keyword): Returns tables and columns matching the keyword.
- get_column_description(table, column): Returns the description for a specific column.
- semantic_search_schema(keyword): Returns top semantically matching schema elements for a keyword (BM25+SBERT fusion).
- relation_aware_retrieval(nl_query, top_k=10): Returns top-k relevant schema elements using graph-based, context-aware retrieval. Use this tool when the query is ambiguous, involves multiple tables, or when other tools do not provide sufficient context.

When you need schema details, respond with a tool call in the format:
TOOL: <tool_name>(<arguments>)

If the user query is ambiguous, incomplete, or you cannot proceed for any reason, you MUST output a CLARIFY statement in the format:
CLARIFY: <your clarification question>
Do NOT issue further tool calls until the user responds to your clarification.
If you can partially synthesize the IR, output a partial IR and ask for clarification on the missing or ambiguous parts.

Once you have all the information, output the final intermediate representation (IR) in JSON, prefixed by:
IR: { "query_understanding": ..., "entities": ..., "metrics": ..., "filters": ..., "time_dimensions": ..., "groupings": ..., "execution_plan": ..., "complete_sql_pseudocode": ... }

The IR should include:
- Query understanding
- Entities (primary and secondary, with join keys if relevant)
- Metrics and measures
- Filter conditions
- Time dimensions
- Groupings
- Step-by-step SQL execution plan
- Final single-line SQL pseudocode (complete_sql_pseudocode) using CTEs with no newline characters

Think step by step, and do not guess about the schema. If you are stuck, always ask the user a clarifying question.

If you are unsure which tables or columns are relevant, or if the query involves complex relationships, consider using relation_aware_retrieval to retrieve the most contextually and relationally relevant schema elements.

***IMPORTANT: If you need clarification from the user, you MUST respond ONLY with 'CLARIFY: <your question>' and nothing else. Do NOT use any other format for clarifications.***
"""

import re
import os
from NL_query_decomposition.schema_tools import SchemaTools
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import argparse
import time
import logging

SYSTEM_PROMPT = """
You are an expert SQL query architect specializing in translating complex natural language analytical questions into structured, machine-readable intermediate representations (IRs) in JSON format.

You have access to the following tools for schema exploration:
- list_tables(): Returns all table names in the database.
- get_columns(table_name): Returns all columns (and their descriptions) for a given table.
- search_schema(keyword): Returns tables and columns matching the keyword.
- get_column_description(table, column): Returns the description for a specific column.
- semantic_search_schema(keyword): Returns top semantically matching schema elements for a keyword (BM25+SBERT fusion).
- relation_aware_retrieval(nl_query, top_k=10): Returns top-k relevant schema elements using graph-based, context-aware retrieval. Use this tool when the query is ambiguous, involves multiple tables, or when other tools do not provide sufficient context.

When you need schema details, respond with a tool call in the format:
TOOL: <tool_name>(<arguments>)

If the user query is ambiguous, incomplete, or you cannot proceed for any reason, you MUST output a CLARIFY statement in the format:
CLARIFY: <your clarification question>
Do NOT issue further tool calls until the user responds to your clarification.
If you can partially synthesize the IR, output a partial IR and ask for clarification on the missing or ambiguous parts.

Once you have all the information, output the final intermediate representation (IR) in JSON, prefixed by:
IR: { "query_understanding": ..., "entities": ..., "metrics": ..., "filters": ..., "time_dimensions": ..., "groupings": ..., "execution_plan": ..., "complete_sql_pseudocode": ... }

The IR should include:
- Query understanding
- Entities (primary and secondary, with join keys if relevant)
- Metrics and measures
- Filter conditions
- Time dimensions
- Groupings
- Step-by-step SQL execution plan
- Final single-line SQL pseudocode (complete_sql_pseudocode) using CTEs with no newline characters

Think step by step, and do not guess about the schema. If you are stuck, always ask the user a clarifying question.

If you are unsure which tables or columns are relevant, or if the query involves complex relationships, consider using relation_aware_retrieval to retrieve the most contextually and relationally relevant schema elements.

***IMPORTANT: If you need clarification from the user, you MUST respond ONLY with 'CLARIFY: <your question>' and nothing else. Do NOT use any other format for clarifications.***
"""

# Load Claude API key from config
global_config = Path(__file__).parent.parent / "global" / "config.env"
load_dotenv(global_config)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class ConversationMemory:
    def __init__(self, window=10):
        self.turns = []  # Each turn: {query, ir, sql, result}
        self.window = window

    def add_turn(self, query, ir=None, sql=None, result=None):
        self.turns.append({"query": query, "ir": ir, "sql": sql, "result": result})
        if len(self.turns) > self.window:
            self.turns = self.turns[-self.window:]

    def get_last_turn(self):
        return self.turns[-1] if self.turns else None

    def get_context_block(self):
        if not self.turns:
            return ""
        last = self.turns[-1]
        context = f"Previous Query: {last['query']}\n"
        if last.get('sql'):
            context += f"Previous SQL: {last['sql']}\n"
        elif last.get('ir'):
            context += f"Previous IR: {last['ir']}\n"
        if last.get('result'):
            context += f"Previous Result: {last['result']}\n"
        return context

# --- LLM-based coreference resolver ---
def resolve_followup_with_llm(previous_query, followup_query, model="claude-3-7-sonnet-20250219", max_tokens=256):
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    global_config = Path(__file__).parent.parent / "global" / "config.env"
    load_dotenv(global_config)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = "You are a helpful assistant that rewrites ambiguous follow-up questions to be explicit, using the context of the previous query."
    user_prompt = f"Previous Query: {previous_query}\nFollow-up Query: {followup_query}\nRewrite the follow-up query to be explicit, using the context of the previous query. Only output the rewritten query."
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        response = message.content
        if isinstance(response, list):
            rewritten = " ".join(getattr(part, 'text', str(part)) for part in response)
        else:
            rewritten = str(response)
        return rewritten.strip()
    except Exception as e:
        print(f"[Coreference LLM ERROR] {e}. Using original follow-up query.")
        return followup_query

class ToolAugmentedAgent:
    def __init__(self, schema_tools=None, model="claude-3-7-sonnet-20250219", max_tokens=1024):
        self.schema_tools = schema_tools or SchemaTools()
        self.context = []  # List of (role, content) tuples
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def add_to_context(self, role, content):
        self.context.append((role, content))

    def get_prompt(self):
        return "\n".join(f"{role.upper()}: {content}" for role, content in self.context if role != 'system')

    def parse_llm_output(self, output):
        output = output.strip()
        # Look for IR anywhere in the output
        ir_match = re.search(r'IR:\s*(\{.*)', output, re.DOTALL)
        if ir_match:
            return ('ir', ir_match.group(1).strip())
        # Look for TOOL call anywhere in the output
        tool_match = re.search(r'TOOL:\s*(\w+)\((.*)\)', output)
        if tool_match:
            tool_name = tool_match.group(1)
            args = tool_match.group(2)
            return ('tool', tool_name, args)
        # Look for CLARIFY anywhere in the output
        clarify_match = re.search(r'CLARIFY:\s*(.*)', output)
        if clarify_match:
            return ('clarify', clarify_match.group(1).strip())
        return ('other', output)

    def execute_tool(self, tool_name, args):
        import ast
        # Helper to extract value from key="value" or just value
        def extract_arg(arg):
            arg = arg.strip()
            # If arg is of the form key="value" or key='value', extract value
            m = re.match(r'\s*\w+\s*=\s*["\'](.+)["\']\s*$', arg)
            if m:
                return m.group(1)
            # If arg is quoted, remove quotes
            if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                return arg[1:-1]
            return arg
        try:
            parsed_args = ast.literal_eval(f'({args})')
        except Exception:
            parsed_args = (args,)
        if not isinstance(parsed_args, tuple):
            parsed_args = (parsed_args,)
        # Extract values for each argument
        parsed_args = tuple(extract_arg(str(a)) for a in parsed_args)
        if tool_name == "list_tables":
            return self.schema_tools.list_tables()
        elif tool_name == "get_columns":
            return self.schema_tools.get_columns(*parsed_args)
        elif tool_name == "search_schema":
            return self.schema_tools.search_schema(*parsed_args)
        elif tool_name == "get_column_description":
            return self.schema_tools.get_column_description(*parsed_args)
        elif tool_name == "semantic_search_schema":
            results = self.schema_tools.semantic_search_schema(*parsed_args)
            # Print results in a readable way
            print("\n[Semantic Search Results]")
            for i, (elem, score) in enumerate(results, 1):
                print(f"{i}. [{score:.2f}] {elem}")
            return results
        elif tool_name == "relation_aware_retrieval":
            # Accepts: (nl_query, top_k)
            if len(parsed_args) == 2:
                return self.schema_tools.relation_aware_retrieval(parsed_args[0], int(parsed_args[1]))
            elif len(parsed_args) == 1:
                return self.schema_tools.relation_aware_retrieval(parsed_args[0])
            else:
                return "relation_aware_retrieval requires at least the NL query as argument."
        else:
            return f"Unknown tool: {tool_name}"

    def call_llm(self, user_query, timeout=30, max_tokens=8192):
        messages = []
        for role, content in self.context:
            if role == 'tool':
                messages.append({"role": "assistant", "content": content})
            elif role == 'user':
                messages.append({"role": "user", "content": content})
            elif role == 'clarify':
                messages.append({"role": "assistant", "content": content})
        if not messages or messages[0]["role"] != "user":
            messages = [{"role": "user", "content": user_query}] + messages
        start_time = time.time()
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages
            )
            duration = time.time() - start_time
            logging.info(f"LLM call completed in {duration:.2f}s")
            response = message.content
            if isinstance(response, list):
                try:
                    response = " ".join(getattr(part, 'text', str(part)) for part in response)
                except Exception:
                    response = str(response)
            print("\n--- RAW CLAUDE RESPONSE ---\n")
            print(response)
            print("\n--- END RAW RESPONSE ---\n")
            return response
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"LLM call failed after {duration:.2f}s: {e}")
            return None

    # --- Progress Tracker ---
    class QueryProgressTracker:
        def __init__(self, user_query):
            self.user_query = user_query
            self.required_elements = self.extract_query_elements(user_query)
            self.mapped_elements = set()
            self.explored_schema = set()  # (tool_name, arg)
            self.ambiguous_elements = set()

        def extract_query_elements(self, query):
            # Simple keyword-based extraction; can be replaced with LLM-based extraction
            # For demo: look for words like 'customer', 'sales', 'date', 'total', etc.
            import re
            tokens = set(re.findall(r'\w+', query.lower()))
            # In practice, use LLM or a more robust parser
            return tokens

        def update_mapping(self, tool_name, args, tool_result):
            # Mark schema elements as explored
            self.explored_schema.add((tool_name, args))
            # Try to map required elements to schema elements
            if tool_name == 'get_columns' and tool_result:
                table = args.split(',')[0].strip("'\" ")
                for col, desc in tool_result:
                    for elem in self.required_elements:
                        if elem in col.lower() or (desc and elem in str(desc).lower()):
                            self.mapped_elements.add(elem)
            elif tool_name == 'search_schema' and tool_result:
                for res in tool_result:
                    for elem in self.required_elements:
                        if elem in (res.get('table','').lower() + res.get('column','').lower()):
                            self.mapped_elements.add(elem)
            elif tool_name == 'semantic_search_schema' and tool_result:
                for elem_score in tool_result:
                    elem, _ = elem_score
                    for req in self.required_elements:
                        if req in elem.lower():
                            self.mapped_elements.add(req)

        def all_mapped(self):
            return self.required_elements <= self.mapped_elements

        def is_stuck(self, prev_explored):
            # If no new schema elements explored since last check, agent is stuck
            return self.explored_schema == prev_explored

        def unmapped_elements(self):
            return self.required_elements - self.mapped_elements

        def add_ambiguous(self, elem):
            self.ambiguous_elements.add(elem)

    def generate_clarification(self, progress_tracker, schema_context, user_query):
        """
        Generate a context-aware clarification question for unmapped/ambiguous elements.
        - Only ask about elements that are truly ambiguous after schema exploration.
        - Suggest plausible interpretations for ambiguous phrases.
        - Use natural language, not token lists.
        """
        unmapped = progress_tracker.unmapped_elements() if hasattr(progress_tracker, 'unmapped_elements') else set()
        ambiguous = progress_tracker.ambiguous_elements if hasattr(progress_tracker, 'ambiguous_elements') else set()
        # Example: if 'product from Canada' is ambiguous, suggest plausible meanings
        clarifications = []
        # Example heuristics for common ambiguous phrases
        if any('product' in e or 'item' in e for e in unmapped | ambiguous):
            clarifications.append("When you say 'product from Canada', do you mean products manufactured in Canada, products purchased in Canadian stores, or products shipped to Canada?")
        if any('user' in e or 'customer' in e for e in unmapped | ambiguous):
            clarifications.append("Do you mean all users/customers, or a specific segment?")
        # Add more heuristics as needed
        # If no specific phrase, fall back to a generic but natural clarification
        if not clarifications and unmapped:
            clarifications.append(f"Could you clarify what you mean by: {', '.join(unmapped)}?")
        if not clarifications:
            clarifications.append("Could you clarify your request?")
        return ' '.join(clarifications)

    def generate_ir(self, user_query, max_turns=12, sql_dialect='trino', max_total_tool_calls=8, ir_timeout=300):
        """
        Generate only the IR (no pseudocode) from the NL query.
        """
        self.context = []
        print(f"\n[Starting IR generation for query: {user_query}]")
        consecutive_tool_calls = 0
        total_tool_calls = 0
        last_llm_output = None
        ambiguous_terms = set()
        partial_ir = None
        final_ir = None
        progress_tracker = self.QueryProgressTracker(user_query)
        prev_explored = set()
        ir_start_time = time.time()
        for turn in range(max_turns):
            if time.time() - ir_start_time > ir_timeout:
                logging.error(f"IR generation timed out after {ir_timeout}s")
                break
            prompt = self.get_prompt()
            # Add instruction to output only IR, no pseudocode
            prompt += "\n[INSTRUCTION: Output only the IR in JSON. Do NOT include SQL pseudocode or explanations.]"
            print(f"\n[Turn {turn+1} | Prompt to LLM:]")
            print(f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {user_query}\n{prompt}")
            llm_output = self.call_llm(user_query, timeout=ir_timeout)
            last_llm_output = llm_output
            if llm_output is None:
                logging.error("LLM call failed or timed out during IR generation.")
                break
            kind, *payload = self.parse_llm_output(llm_output)
            if kind == 'tool':
                consecutive_tool_calls += 1
                total_tool_calls += 1
                tool_name, args = payload
                print(f"[Executing tool: {tool_name}({args})]")
                tool_result = self.execute_tool(tool_name, args)
                self.add_to_context('tool', f"{tool_name}({args}) => {tool_result}")
                progress_tracker.update_mapping(tool_name, args, tool_result)
                if progress_tracker.all_mapped():
                    print("[All required query elements mapped. Prompting LLM to synthesize IR...]")
                    self.add_to_context('tool', '[INFO] All required elements mapped. Please synthesize the IR now.')
                    continue
                if progress_tracker.is_stuck(prev_explored):
                    unmapped = progress_tracker.unmapped_elements()
                    if progress_tracker.mapped_elements:
                        partial_ir_dict = {
                            "query_understanding": user_query,
                            "entities": list(progress_tracker.mapped_elements),
                            "metrics": [],
                            "filters": [],
                            "time_dimensions": [],
                            "groupings": [],
                            "execution_plan": []
                        }
                        print("[Partial IR synthesized so far:]")
                        print(partial_ir_dict)
                    clarify_msg = self.generate_clarification(progress_tracker, schema_context=None, user_query=user_query)
                    user_input = input(f"CLARIFY: {clarify_msg}\nYour answer: ")
                    self.add_to_context('user', user_input)
                    consecutive_tool_calls = 0
                    total_tool_calls = 0
                    ambiguous_terms.clear()
                    prev_explored = set(progress_tracker.explored_schema)
                    continue
                prev_explored = set(progress_tracker.explored_schema)
                if total_tool_calls >= max_total_tool_calls:
                    unmapped = progress_tracker.unmapped_elements()
                    if progress_tracker.mapped_elements:
                        partial_ir_dict = {
                            "query_understanding": user_query,
                            "entities": list(progress_tracker.mapped_elements),
                            "metrics": [],
                            "filters": [],
                            "time_dimensions": [],
                            "groupings": [],
                            "execution_plan": []
                        }
                        print("[Partial IR synthesized so far:]")
                        print(partial_ir_dict)
                    print("[Global circuit breaker: Too many total tool calls without IR synthesis.]")
                    clarify_msg = "I am unable to fully resolve your query after multiple schema explorations."
                    if unmapped:
                        clarify_msg += f" The following terms are ambiguous or not found in the schema: {', '.join(unmapped)}."
                    clarify_msg += " Could you clarify or provide more details?"
                    user_input = input(f"CLARIFY: {clarify_msg}\nYour answer: ")
                    self.add_to_context('user', user_input)
                    consecutive_tool_calls = 0
                    total_tool_calls = 0
                    ambiguous_terms.clear()
                if consecutive_tool_calls >= 3:
                    unmapped = progress_tracker.unmapped_elements()
                    if progress_tracker.mapped_elements:
                        partial_ir_dict = {
                            "query_understanding": user_query,
                            "entities": list(progress_tracker.mapped_elements),
                            "metrics": [],
                            "filters": [],
                            "time_dimensions": [],
                            "groupings": [],
                            "execution_plan": []
                        }
                        print("[Partial IR synthesized so far:]")
                        print(partial_ir_dict)
                    print("[Agent circuit breaker: Too many consecutive tool calls without progress.]")
                    clarify_msg = "I am unable to fully resolve your query."
                    if unmapped:
                        clarify_msg += f" The following terms are ambiguous or not found in the schema: {', '.join(unmapped)}."
                    clarify_msg += " Could you clarify or provide more details?"
                    user_input = input(f"CLARIFY: {clarify_msg}\nYour answer: ")
                    self.add_to_context('user', user_input)
                    consecutive_tool_calls = 0
                    ambiguous_terms.clear()
                continue
            elif kind == 'clarify':
                consecutive_tool_calls = 0
                clarification = payload[0]
                print(f"[LLM requests clarification: {clarification}]")
                user_input = input(f"Clarification needed: {clarification}\nYour answer: ")
                self.add_to_context('user', user_input)
                continue
            elif kind == 'ir':
                print("\n[Final IR produced by LLM (no SQL or pseudocode):]")
                ir_text = payload[0]
                import json
                try:
                    ir_json = json.loads(ir_text)
                    final_ir = ir_json
                except Exception:
                    final_ir = ir_text
                print(final_ir)
                logging.info("IR successfully produced. Proceed to pseudocode generation step.")
                return final_ir
            else:
                consecutive_tool_calls = 0
                ir_match = re.search(r'IR:\s*(\{.*)', llm_output, re.DOTALL)
                if ir_match:
                    partial_ir = ir_match.group(1).strip()
                    print("[Partial IR detected in LLM output]")
                    print(partial_ir)
                print(f"[LLM output not recognized: {llm_output}]")
                if total_tool_calls >= max_total_tool_calls:
                    print("[Global circuit breaker: Too many total tool calls without IR synthesis.]")
                    clarify_msg = "I am unable to fully resolve your query after multiple schema explorations."
                    if ambiguous_terms:
                        clarify_msg += f" The following terms are ambiguous or not found in the schema: {', '.join(ambiguous_terms)}."
                    clarify_msg += " Could you clarify or provide more details?"
                    if partial_ir:
                        print("[Partial IR synthesized so far:]")
                        print(partial_ir)
                    user_input = input(f"CLARIFY: {clarify_msg}\nYour answer: ")
                    self.add_to_context('user', user_input)
                    total_tool_calls = 0
                    ambiguous_terms.clear()
                continue
        print("[Agent loop ended without producing IR]")
        if partial_ir:
            print("[Partial IR synthesized so far:]")
            print(partial_ir)
        print(f"[Last LLM output: {last_llm_output}")
        if not final_ir:
            if 'progress_tracker' in locals():
                clarification = self.generate_clarification(progress_tracker, schema_context=None, user_query=user_query)
                print(f"CLARIFY: {clarification}")
        return None

    def generate_sql_pseudocode(self, ir, model="claude-3-7-sonnet-20250219", max_tokens=8192):
        """
        Given an IR, generate only the SQL pseudocode (single-line, CTEs, etc.) using the LLM.
        """
        import anthropic
        import os
        import json
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        ir_json = json.dumps(ir, indent=2)
        prompt = f"""
Given the following intermediate representation (IR) of a query, output only the SQL pseudocode (single-line, CTEs, etc.). Do NOT include explanations or extra text.\n\nIR:\n{ir_json}\n"""
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system="You are an expert SQL query architect. Output only the SQL pseudocode for the given IR.",
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content
            if isinstance(response, list):
                sql_pseudocode = " ".join(getattr(part, 'text', str(part)) for part in response)
            else:
                sql_pseudocode = str(response)
            sql_pseudocode = sql_pseudocode.strip()
            if sql_pseudocode.startswith("```sql"):
                sql_pseudocode = sql_pseudocode[6:]
            if sql_pseudocode.endswith("```"):
                sql_pseudocode = sql_pseudocode[:-3]
            print("\n[SQL pseudocode generated from IR:]")
            print(sql_pseudocode)
            return sql_pseudocode
        except Exception as e:
            print(f"-- ERROR: Failed to generate SQL pseudocode from IR: {e}")
            return None

class ToolAugmentedAgentWithMemory(ToolAugmentedAgent):
    def __init__(self, schema_tools=None, model="claude-3-7-sonnet-20250219", max_tokens=1024, memory_window=10):
        super().__init__(schema_tools=schema_tools, model=model, max_tokens=max_tokens)
        self.memory = ConversationMemory(window=memory_window)

    def get_prompt(self):
        memory_block = self.memory.get_context_block()
        context_part = super().get_prompt()
        if memory_block:
            return memory_block + "\n" + context_part
        return context_part

    def generate_ir(self, user_query, max_turns=12, sql_dialect='trino', max_total_tool_calls=8, ir_timeout=300):
        # Use LLM-based coreference resolution if there is a previous turn
        if len(self.memory.turns) > 0:
            previous_query = self.memory.get_last_turn()['query']
            explicit_query = resolve_followup_with_llm(previous_query, user_query)
        else:
            explicit_query = user_query
        self.memory.add_turn(explicit_query)  # Add to memory before IR
        self.context = []  # Reset context for new turn
        ir = super().generate_ir(explicit_query, max_turns, sql_dialect, max_total_tool_calls, ir_timeout)
        self.memory.turns[-1]['ir'] = ir
        return ir

    def generate_sql_pseudocode(self, ir, model="claude-3-7-sonnet-20250219", max_tokens=8192):
        sql_pseudocode = super().generate_sql_pseudocode(ir, model, max_tokens)
        if self.memory.turns:
            self.memory.turns[-1]['sql'] = sql_pseudocode
        return sql_pseudocode

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tool-augmented NL2SQL agent with structured conversational memory.")
    parser.add_argument('--query', '-q', type=str, help='Natural language query to decompose.')
    args = parser.parse_args()
    agent = ToolAugmentedAgentWithMemory()
    while True:
        if args.query:
            nl_query = args.query
            args.query = None  # Only use once
        else:
            nl_query = input("Enter a natural language query (or 'exit' to quit): ")
            if nl_query.strip().lower() == 'exit':
                break
        ir = agent.generate_ir(nl_query, ir_timeout=300)
        if ir and isinstance(ir, dict):
            from sql_generator import generate_sql_from_ir
            from sql_generator import prune_ir
            import time
            import logging
            sql_start = time.time()
            try:
                sql_pseudocode = agent.generate_sql_pseudocode(ir)
                concise_ir = prune_ir(ir)
                sql_code = generate_sql_from_ir(concise_ir, dialect='trino', sql_pseudocode=sql_pseudocode, nl_query=nl_query)
                sql_duration = time.time() - sql_start
                logging.info(f"SQL generation completed in {sql_duration:.2f}s")
                print("\n[SQL generated from pseudocode and concise IR:]")
                print(sql_code)
                # Update memory with SQL
                if agent.memory.turns:
                    agent.memory.turns[-1]['sql'] = sql_code
                # Validate SQL with SQLGlot
                import sqlglot
                try:
                    parsed = sqlglot.parse_one(sql_code, read='trino')
                    formatted_sql = parsed.sql(pretty=True)
                    print("\n[SQLGlot validation: SQL is valid. Formatted SQL:]")
                    print(formatted_sql)
                except Exception as e:
                    print("\n[SQLGlot validation: SQL is INVALID! Error:]")
                    print(e)
                    print("[Propagating error back to agent/user]")
            except Exception as e:
                sql_duration = time.time() - sql_start
                logging.error(f"SQL generation failed after {sql_duration:.2f}s: {e}")
                print("[ERROR] SQL generation failed. See logs for details.")
        else:
            print("[No valid IR produced. SQL pseudocode and SQL generation skipped.]") 