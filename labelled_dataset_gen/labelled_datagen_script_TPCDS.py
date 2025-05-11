from pathlib import Path
import sys
import openai
import json
import os
import time
import random
import re # Added for JSON cleanup
from collections import defaultdict
from datetime import datetime, timedelta # Added for timestamp

# --- Path Setup (similar to original script) ---
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent.parent
project_root = src_dir.parent
sys.path.append(str(src_dir))

# Assuming sqlglot is installed: pip install sqlglot
try:
    from sqlglot import parse_one
    # Use alias for sqlglot.expressions to avoid potential name conflicts if user adds other vars
    import sqlglot.expressions as sqlglot_exp
    # Import the qualify function from sqlglot.optimizer
    from sqlglot.optimizer import qualify
    from sqlglot.errors import OptimizeError, ParseError
except ImportError as e:
    print(f"ERROR: Failed to import sqlglot components. Is sqlglot installed correctly? Error: {e}")
    exit() # Exit if imports fail

# --- Configuration ---
try:
    from config.settings import settings
    schema_path_str = settings.neo4j.schema_profile_file
    query_path_str = settings.neo4j.query_dataset_file
    SCHEMA_FILENAME = Path(schema_path_str) if os.path.isabs(schema_path_str) else project_root / schema_path_str
    OUTPUT_FILENAME = Path(query_path_str) if os.path.isabs(query_path_str) else project_root / query_path_str
    LLM_MODEL = settings.openai.model
    OPENAI_API_KEY = settings.openai.api_key
except ImportError as e:
    print(f"Error: Could not import settings for schema and query files and OPENAI API key: {e}")
    LLM_MODEL = "o3-mini" # Model for generation
    OPENAI_API_KEY = None
SQL_DIALECT = "snowflake" # Dialect for sqlglot parsing/qualification
# --- End Configuration ---


# Initialize the OpenAI client interface (ensure API key is set in environment)
try:
    # Ensure API key is set via environment variable:
    # export OPENAI_API_KEY='your-key-here'
    client = openai.Client(api_key=OPENAI_API_KEY)
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        # Consider adding 'client = None' if you want the script to fail gracefully later
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client. Error: {e}")
    client = None # Set client to None if init fails


# --- Schema Loading Function ---
def load_schema_from_json(schema_file):
    """ Loads schema from the specified JSON file into sqlglot format. """
    if not os.path.exists(schema_file):
        print(f"ERROR: Schema file not found: {schema_file}")
        return None
    print(f"Loading schema from: {schema_file}")
    try:
        with open(schema_file, 'r') as f:
            raw_schema_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to read or parse schema file {schema_file}. Error: {e}")
        return None
    if 'tables' not in raw_schema_data:
        print(f"ERROR: Expected 'tables' key in {schema_file}")
        return None
    sqlglot_schema = {}
    for table_name, table_info in raw_schema_data['tables'].items():
        if 'columns' in table_info:
            sqlglot_schema[table_name] = {
                col_name: col_info.get('type', 'UNKNOWN')
                for col_name, col_info in table_info['columns'].items()
            }
    print(f"Successfully loaded schema for {len(sqlglot_schema)} tables.")
    return sqlglot_schema

# --- Table/Column Extraction Function (Final Working Version) ---
def extract_qualified_tables_and_columns(sql_query, schema, dialect="snowflake"):
    """
    Extracts base tables and their columns from a SQL query using sqlglot.optimizer.qualify
    to resolve unqualified columns based on the provided schema.
    Returns: (tables_list, columns_dict, had_error)
    """
    if not schema: return [], {}, True
    if not sql_query: return [], {}, False

    try:
        expression = parse_one(sql_query, read=dialect)
        qualified_expression = qualify.qualify(
            expression, schema=schema, dialect=dialect,
            validate_qualify_columns=False, identify=False
        )

        tables = set()
        table_aliases = {}
        table_columns = defaultdict(set)
        cte_names = set()

        # Use sqlglot_exp alias here
        for cte in qualified_expression.find_all(sqlglot_exp.CTE):
            cte_names.add(cte.alias_or_name)

        for table_expr in qualified_expression.find_all(sqlglot_exp.Table):
            # Check if it exists in schema and is not a CTE
            if table_expr.name in schema and table_expr.name not in cte_names:
                original_table_name = table_expr.name
                tables.add(original_table_name)
                if table_expr.alias:
                    table_aliases[table_expr.alias] = original_table_name

        for col in qualified_expression.find_all(sqlglot_exp.Column):
            table_ref = col.table
            column_name = col.name
            if table_ref in table_aliases:
                original_table_name = table_aliases[table_ref]
                if original_table_name in tables: # Check if alias maps to a base table
                    table_columns[original_table_name].add(column_name)
            elif table_ref in tables: # Check if direct ref is a base table
                table_columns[table_ref].add(column_name)

        final_map = {tbl: sorted(list(cols)) for tbl, cols in table_columns.items()}
        for tbl in tables: # Ensure all identified tables are keys
            if tbl not in final_map: final_map[tbl] = []

        return sorted(list(tables)), final_map, False # No error

    except (OptimizeError, ParseError, Exception) as e:
        # Log error during extraction attempt
        # print(f"WARN: Error during extraction for SQL: {sql_query[:100]}... Error: {e}")
        return [], {}, True # Indicate error occurred


# --- Constants (Copied from original script) ---
TPCDS_TABLES = [
    "store_sales", "store_returns", "catalog_sales", "catalog_returns",
    "web_sales", "web_returns", "inventory", "store", "call_center",
    "catalog_page", "web_site", "web_page", "warehouse", "customer",
    "customer_address", "customer_demographics", "date_dim", "time_dim",
    "item", "income_band", "household_demographics", "reason", "ship_mode",
    "promotion"
]
BUSINESS_CATEGORIES = [
    "Customer Segmentation and Behavior Analysis",
    "Product Performance and Inventory Analysis",
    "Sales and Revenue Analysis",
    "Marketing Campaign Effectiveness",
    "Return Rate and Customer Satisfaction Analysis",
    "Seasonal and Temporal Trends",
    "Predictive Analytics and Forecasting",
    "Cross-channel Performance Comparison",
    "Operational Efficiency Analysis",
    "Cohort Analysis and Customer Lifetime Value"
]
SQL_FEATURES = [
    "Common Table Expressions (CTEs)", "Window Functions", "Nested Subqueries",
    "Complex Aggregations", "Conditional Logic (CASE statements)", "Pivot Operations",
    "Self-Joins", "Date-based Analysis Functions", "Set Operations (UNION, INTERSECT, EXCEPT)",
    "Multi-level Joins"
]
COMPLEXITY_LEVELS = {
    "basic": {"description": "Straightforward...", "percentage": 15, "min_features": 0, "max_features": 1},
    "standard": {"description": "Regular...", "percentage": 40, "min_features": 1, "max_features": 2},
    "advanced": {"description": "Complex...", "percentage": 30, "min_features": 2, "max_features": 4},
    "expert": {"description": "Sophisticated...", "percentage": 15, "min_features": 3, "max_features": 6}
}
PERSONAS = [ # Not used in generation logic, kept for reference
    {"name": "Data Engineer", "focus": "...", "style": "..."},
    {"name": "Business Analyst", "focus": "...", "style": "..."},
    {"name": "Executive", "focus": "...", "style": "..."},
    {"name": "Data Scientist", "focus": "...", "style": "..."}
]


# --- Updated Query Generation Function ---
def generate_query_batch(batch_size, complexity_level, loaded_schema, business_category=None, focus_tables=None):
    """
    Generates a batch of SQL queries using an LLM, providing schema context
    and requesting tables_used in the prompt.
    """
    if not client: print("ERROR: OpenAI client not initialized."); return []
    if not loaded_schema: print("ERROR: Schema not loaded."); return []

    # Build Schema String for Prompt
    schema_prompt_string = "TPC-DS Schema Definition:\n"
    tables_to_include_in_prompt = focus_tables if focus_tables else loaded_schema.keys()
    included_count = 0
    MAX_TABLES_IN_PROMPT = 15
    for table in tables_to_include_in_prompt:
        if table in loaded_schema:
            schema_prompt_string += f"Table: {table}\n Columns:\n"
            col_defs = [f"  - {name} ({ctype})" for name, ctype in loaded_schema[table].items()]
            schema_prompt_string += "\n".join(col_defs) + "\n\n"
            included_count += 1
            if included_count >= MAX_TABLES_IN_PROMPT and focus_tables is None:
                schema_prompt_string += "(Schema truncated for brevity...)\n\n"; break
    if not schema_prompt_string.endswith("\n\n"): schema_prompt_string += "\n"

    # Prompt Enhancements
    schema_adherence_instruction = (
        "VERY IMPORTANT: You MUST strictly adhere to the TPC-DS Schema Definition provided above. "
        "ONLY use the tables listed. For each table, ONLY use the columns listed under it. "
        "Do NOT invent columns or use columns that are not explicitly listed for a table. "
        "Ensure all column references are valid for the tables they are used with."
    )
    # *** UPDATED JSON format instruction to RE-INCLUDE tables_used ***
    json_format_instruction = (
         "Format each query object with these exact keys:\n"
        "- 'sql': A valid SQL query strictly adhering to the provided TPC-DS schema.\n"
        "- 'business_question': A clear explanation of the specific business question/problem the query answers.\n"
        "- 'natural_language': A detailed explanation of what the SQL query does in natural language.\n"
        "- 'alternatives': A list of 3 alternative explanations from different perspectives.\n"
        "- 'tables_used': List of all TPC-DS base tables referenced in the query (derived from your generated SQL).\n" # Requested from LLM
        "- 'sql_features': List of advanced SQL features used in the query.\n"
        "- 'complexity_level': The complexity level of this query (basic, standard, advanced, or expert).\n\n"
         # NOTE: columns_used is NOT requested from LLM, will be added via sqlglot
    )

    # Complexity, Category, Focus instructions (using full definitions from user script)
    complexity_info = COMPLEXITY_LEVELS[complexity_level]
    min_features = complexity_info["min_features"]
    max_features = complexity_info["max_features"]
    if complexity_level == "basic": complexity_instruction = f"Generate BASIC business queries that:\n- Use {min_features}-{max_features} advanced SQL features...\n- Are typical of daily operational reporting" # Truncated for brevity
    elif complexity_level == "standard": complexity_instruction = f"Generate STANDARD business queries that:\n- Use {min_features}-{max_features} advanced SQL features...\n- Include some calculations and filtering logic" # Truncated
    elif complexity_level == "advanced": complexity_instruction = f"Generate ADVANCED business queries that:\n- Use {min_features}-{max_features} advanced SQL features...\n- Include advanced calculations and business logic" # Truncated
    else: complexity_instruction = f"Generate EXPERT-LEVEL business queries that:\n- Use {min_features}-{max_features} advanced SQL features...\n- Represent the kind of analysis done by experienced data analysts" # Truncated
    category_instruction = f"Focus on the business category: {business_category}. " if business_category else ""
    table_focus_instr = f"Ensure these tables are incorporated if relevant: {', '.join(focus_tables)}. " if focus_tables else ""

    # Include other prompt parts from user script (not truncated now)
    business_context = (
        "Each query MUST solve a realistic business problem that a retail or e-commerce "
        "company would actually encounter. The queries should provide actionable insights "
        "that would inform business decisions and strategy."
    )
    general_requirement = (
        "GENERAL REQUIREMENTS:\n"
        "1. Every query must have clear business value and context\n"
        "2. Simple queries should still answer meaningful business questions\n"
        "3. Don't generate meaningless queries like 'SELECT ss_store_sk, COUNT(*) FROM store_sales GROUP BY ss_store_sk'\n"
        "4. Even simple queries should provide useful business information\n"
        "5. Queries should be realistic for their complexity level"
    )
    interpretation_instructions = (
        "INTERPRETATION REQUIREMENTS:\n"
        "1. For each query, provide a detailed 'natural_language' explanation that accurately describes what the SQL query does\n"
        "2. Also provide three alternative explanations in the 'alternatives' field that explain the same query from different perspectives:\n"
        "   - Technical perspective (focusing on the SQL operations)\n"
        "   - Business perspective (focusing on business value and insights)\n"
        "   - Executive perspective (focusing on high-level strategic implications)\n"
        "3. Each alternative should be clear enough for someone without SQL knowledge to understand"
    )
    example_scenarios = ( # Kept full examples
        "EXAMPLE BUSINESS SCENARIOS (by complexity):\n\n"
        "BASIC:\n- Total monthly sales by product category\n- Customer count by state\n- Return rate by product department\n\n"
        "STANDARD:\n- Year-over-year sales growth by store\n- Customer retention rate by demographic segment\n- Seasonal sales patterns by product category\n\n"
        "ADVANCED:\n- Customer lifetime value calculation\n- Cross-channel conversion analysis\n- Product affinity analysis\n\n"
        "EXPERT:\n- Predictive churn analysis with customer segmentation\n- Multi-touch attribution modeling\n- Price elasticity analysis with market basket insights"
    )
    sql_feature_examples = ( # Kept full examples
        "ADVANCED SQL FEATURES (reference):\n"
        "- Window functions: ROW_NUMBER(), RANK(), LAG(), LEAD()\n- CTEs for multi-step transformations\n"
        "- Complex CASE statements with business logic\n- Analytic functions and aggregations\n"
        "- Date-based calculations and comparisons\n- Self-joins for hierarchical analysis\n"
        "- Nested subqueries for filtered aggregations\n- Set operations for combining result sets"
    )


    # Assemble the Final Prompt
    prompt = (
        f"Generate a JSON list containing {batch_size} unique, business-oriented query objects for the TPC-DS dataset ({SQL_DIALECT} dialect preferred).\n\n"
        f"--- SCHEMA START ---\n{schema_prompt_string}--- SCHEMA END ---\n\n"
        f"{schema_adherence_instruction}\n\n"
        f"--- QUERY REQUIREMENTS START ---\n"
        f"Complexity: {complexity_instruction}\n"
        f"{category_instruction}{table_focus_instr}\n"
        f"{business_context}\n\n"
        f"{general_requirement}\n\n"
        f"{interpretation_instructions}\n\n"
        f"{example_scenarios}\n\n"
        f"{sql_feature_examples}\n\n"
        f"--- QUERY REQUIREMENTS END ---\n\n"
        f"--- OUTPUT FORMAT START ---\n{json_format_instruction}"
        f"IMPORTANT: Also include the 'tables_used' key, listing the base tables you used from the schema.\n"
        f"--- OUTPUT FORMAT END ---\n\n"
        "Return ONLY the valid JSON array conforming to the specified keys."
    )

    # LLM Call (same as before)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": f"You are an expert BI analyst generating {SQL_DIALECT} TPC-DS queries. Strictly adhere to the provided schema. Include the 'tables_used' key in your JSON output."},
                {"role": "user", "content": prompt}
            ],
           reasoning_effort="high"
            # reasoning_effort="high" # Assuming this is valid for your client/model
        )
        result_text = response.choices[0].message.content.strip()
        # JSON parsing (same as before)
        try:
            data = json.loads(result_text)
            if not isinstance(data, list): data = []
        except json.JSONDecodeError:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result_text, re.DOTALL)
            if json_match:
                try: data = json.loads(json_match.group(0)); assert isinstance(data, list)
                except: data = []
            else: data = []
            if not data: print(f"WARN: LLM response parsing failed. Snippet: {result_text[:200]}...")

    except Exception as e:
        print(f"ERROR: Error during LLM query generation batch: {e}")
        data = []

    return data

# --- Assess Query Quality (from user script) ---
def assess_query_quality(query, complexity_level):
    # Using user's provided assessment logic (commented out parts included for completeness)
    # complexity_info = COMPLEXITY_LEVELS[complexity_level]
    # business_question = query.get('business_question', '')
    # if len(business_question) < 10 or "business" not in business_question.lower(): return False
    # if 'natural_language' not in query or 'alternatives' not in query: return False
    # if not isinstance(query.get('alternatives', []), list) or len(query.get('alternatives', [])) < 3: return False
    # sql_lines = query['sql'].strip().split('\n')
    # min_lines = {"basic": 2, "standard": 5, "advanced": 10, "expert": 15}
    # if len(sql_lines) < min_lines[complexity_level]: return False
    # if complexity_level == "basic":
    #     if query['sql'].lower().count('select') == 1 and query['sql'].lower().count('group by') == 1:
    #         if query['sql'].lower().count('where') == 0 and query['sql'].lower().count('case') == 0: return False
    # if complexity_level != "basic":
    #     sql_text = query['sql'].lower()
    #     advanced_features = 0
    #     if any(feature in sql_text for feature in ['over (', 'row_number()', 'rank()', 'lag(', 'lead(']): advanced_features += 1
    #     if 'with ' in sql_text and ' as (' in sql_text: advanced_features += 1
    #     join_count = sql_text.count(' join ')
    #     if join_count >= 2: advanced_features += 1
    #     if sql_text.count('select') > 1: advanced_features += 1
    #     if sql_text.count('case when') > 0: advanced_features += 1
    #     if advanced_features < complexity_info["min_features"]: return False
    # return True
    # Using simplified check + check for tables_used from LLM now
    return isinstance(query, dict) and \
           'sql' in query and query['sql'] and \
           'natural_language' in query and \
           'alternatives' in query and isinstance(query.get('alternatives'), list) and len(query['alternatives']) >= 3 and \
           'tables_used' in query # Check if LLM included the requested key


# --- Track Table Coverage (Uses 'base_tables' extracted by sqlglot) ---
def track_table_coverage(all_queries):
    covered_tables = set()
    for query in all_queries:
        if 'base_tables' in query and isinstance(query['base_tables'], list):
             covered_tables.update(query['base_tables'])
        elif 'tables_used' in query and isinstance(query['tables_used'], list): # Fallback
            covered_tables.update(query['tables_used'])
    return covered_tables

# --- Main Generation and Enhancement Loop ---
def main():
    if not client: print("Aborting: OpenAI client not initialized."); return

    loaded_schema = load_schema_from_json(SCHEMA_FILENAME)
    if not loaded_schema: print("Aborting: Schema could not be loaded."); return

    all_enhanced_queries = []
    total_queries_target = 3 # Target number
    total_queries_generated_valid = 0
    extraction_errors = 0
    llm_generation_failures = 0 # Counter for empty batches

    # --- Preserving User's Full Complexity/Category Logic ---
    complexity_counts = {}
    for level, info in COMPLEXITY_LEVELS.items():
        complexity_counts[level] = int(total_queries_target * info["percentage"] / 100)
    total_allocated = sum(complexity_counts.values()); diff = total_queries_target - total_allocated
    if diff != 0: complexity_counts["standard"] += diff

    print(f"Planning to generate ~{total_queries_target} queries...")
    current_time = datetime.now() # Initialize timestamp base

    for complexity_level, target_count in complexity_counts.items():
        queries_per_category = max(1, target_count // len(BUSINESS_CATEGORIES))
        remaining = target_count - (queries_per_category * len(BUSINESS_CATEGORIES))
        print(f"\nGenerating {complexity_level} queries (Target: {target_count}):")
        level_generated_count = 0
        processed_categories_for_level = 0

        # Iterate through categories as in user's original script
        for category in BUSINESS_CATEGORIES:
            if level_generated_count >= target_count: break # Stop if target already met

            category_target_for_batch = queries_per_category + (1 if remaining > 0 else 0)
            if remaining > 0: remaining -= 1
            if category_target_for_batch == 0: continue

            print(f"  Category: {category} (Target for this category: {category_target_for_batch})")
            category_generated_count = 0
            category_attempts = 0
            MAX_CATEGORY_ATTEMPTS = category_target_for_batch * 3 # Limit attempts per category

            while category_generated_count < category_target_for_batch and category_attempts < MAX_CATEGORY_ATTEMPTS:
                 if level_generated_count >= target_count: break # Check overall target again

                 category_attempts += 1
                 batch_size = min(5, category_target_for_batch - category_generated_count)
                 if batch_size <= 0: break # Should not happen if loop condition is right

                 # Table focusing logic (from user script)
                 focus_tables = None
                 if total_queries_generated_valid > 50 and category_attempts % 3 == 0 : # Check roughly every 3rd attempt within category
                    covered = track_table_coverage(all_enhanced_queries)
                    uncovered = [t for t in loaded_schema.keys() if t not in covered]
                    if uncovered: focus_tables = random.sample(uncovered, min(3, len(uncovered))); print(f"    Focusing on uncovered tables: {focus_tables}")

                 print(f"    Generating batch of {batch_size} (Attempt {category_attempts})...")

                 batch = generate_query_batch(
                     batch_size, complexity_level, loaded_schema, category, focus_tables
                 )

                 if not batch:
                     llm_generation_failures += batch_size
                     print("    LLM returned empty batch, sleeping...")
                     time.sleep(random.uniform(3, 7))
                     continue

                 # Process batch
                 added_in_batch = 0
                 for query_obj in batch:
                     if level_generated_count >= target_count: break # Check target again

                     if assess_query_quality(query_obj, complexity_level):
                         sql_query = query_obj['sql']
                         base_tables, table_columns_map, had_error = extract_qualified_tables_and_columns(
                             sql_query, loaded_schema, dialect=SQL_DIALECT
                         )

                         if had_error:
                             extraction_errors += 1
                             print(f"    - WARN: Failed extraction for generated query. Discarding.")
                             continue # Skip
                         else:
                             query_obj['base_tables'] = base_tables
                             all_columns = set()
                             for columns in table_columns_map.values(): all_columns.update(columns)
                             query_obj['columns_used'] = sorted(list(all_columns))

                             time_offset = timedelta(seconds=total_queries_generated_valid * 120 + random.randint(0, 7200)) # Varied offset seconds
                             query_timestamp = current_time + time_offset
                             query_obj['timestamp'] = query_timestamp.strftime("%Y-%m-%d %H:%M:%S")

                             # Optional comparison print
                             # if total_queries_generated_valid < 5: ...

                             all_enhanced_queries.append(query_obj)
                             level_generated_count += 1
                             category_generated_count += 1
                             total_queries_generated_valid += 1
                             added_in_batch += 1
                     else:
                         print(f"    - Discarded low-quality generated query.")

                 if added_in_batch > 0:
                      print(f"    + Added {added_in_batch} valid queries for category '{category}'.")

                 time.sleep(random.uniform(2, 5)) # Wait between batches

            if category_generated_count < category_target_for_batch:
                 print(f"  WARN: Did not reach target for category '{category}'. Got {category_generated_count}/{category_target_for_batch}")


        if level_generated_count < target_count:
            print(f"WARN: Did not reach target for {complexity_level}. Generated {level_generated_count}/{target_count}")
            # Optionally add logic here to try generating more for this level if needed


    # --- Check for missing tables and generate targeted queries if needed (from user script) ---
    final_coverage = track_table_coverage(all_enhanced_queries)
    missing_tables = set(loaded_schema.keys()) - final_coverage # Use schema keys

    if missing_tables:
        print(f"\nAttempting to generate additional queries for {len(missing_tables)} missing tables:")
        print(missing_tables)

        for table in list(missing_tables):
            if total_queries_generated_valid >= total_queries_target * 1.1: # Stop if we overshoot too much
                 print("Stopping missing table generation as query count is already high.")
                 break
            print(f"  Focusing on table: {table}")
            complexity = random.choice(["basic", "standard"])
            extra_batch = generate_query_batch(3, complexity, loaded_schema, focus_tables=[table])
            added_for_table = 0
            if extra_batch:
                 for query_obj in extra_batch:
                     if assess_query_quality(query_obj, complexity):
                         sql_query = query_obj['sql']
                         base_tables, table_columns_map, had_error = extract_qualified_tables_and_columns(
                             sql_query, loaded_schema, dialect=SQL_DIALECT
                         )
                         if not had_error and table in base_tables: # Check if extraction worked AND table is present
                              all_columns = set().union(*table_columns_map.values())
                              query_obj['base_tables'] = base_tables
                              query_obj['columns_used'] = sorted(list(all_columns))
                              time_offset = timedelta(seconds=total_queries_generated_valid * 120 + random.randint(0, 7200))
                              query_timestamp = current_time + time_offset
                              query_obj['timestamp'] = query_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                              all_enhanced_queries.append(query_obj)
                              total_queries_generated_valid += 1
                              added_for_table += 1
                              print(f"    + Added query {total_queries_generated_valid} specifically for missing table '{table}'.")
                              break # Only need one query per missing table for basic coverage
            if added_for_table == 0:
                 print(f"    - Could not generate valid query for missing table '{table}'.")
            time.sleep(3)

    # --- Final Save ---
    print(f"\nGenerated {total_queries_generated_valid} total valid & enhanced queries.")
    print(f"Encountered {extraction_errors} errors during table/column extraction (discarded).")
    print(f"LLM generation potentially failed for ~{llm_generation_failures} queries (empty batches).")
    print(f"Saving enhanced queries to {OUTPUT_FILENAME}...")
    try:
        with open(OUTPUT_FILENAME, "w") as outfile:
            json.dump(all_enhanced_queries, outfile, indent=2)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR: Failed to save output JSON file: {e}")

    # --- Final Summary Stats (Copied from user script) ---
    tables_coverage = track_table_coverage(all_enhanced_queries)
    complexity_distribution = {}
    for query in all_enhanced_queries:
        level = query.get('complexity_level', 'unknown')
        complexity_distribution[level] = complexity_distribution.get(level, 0) + 1

    print("\n--- Final Summary ---")
    print("\nTable Coverage:")
    print(f"  {len(tables_coverage)}/{len(loaded_schema)} tables covered ({len(tables_coverage)/len(loaded_schema)*100:.1f}%)")
    missing = set(loaded_schema.keys()) - tables_coverage
    if missing: print(f"  Missing tables: {missing}")

    print("\nComplexity Distribution:")
    for level in COMPLEXITY_LEVELS.keys():
        count = complexity_distribution.get(level, 0)
        percentage = count / total_queries_generated_valid * 100 if total_queries_generated_valid else 0
        print(f"  {level}: {count} queries ({percentage:.1f}%)")

    length_by_complexity = {}
    for query in all_enhanced_queries:
        level = query.get('complexity_level', 'unknown')
        lines = len(query.get('sql','').strip().split('\n'))
        if level not in length_by_complexity: length_by_complexity[level] = []
        length_by_complexity[level].append(lines)

    print("\nAverage Query Length by Complexity:")
    for level, lengths in length_by_complexity.items():
        avg = sum(lengths) / len(lengths) if lengths else 0
        print(f"  {level}: {avg:.1f} lines")

    category_counts = {}
    for query in all_enhanced_queries:
        business_question = query.get('business_question', '').lower()
        matched = False
        for category in BUSINESS_CATEGORIES:
            # Use a simple check for category name keywords
            keywords = category.split(" ")[0:2] # Check first two words maybe
            if all(k.lower() in business_question for k in keywords):
                category_counts[category] = category_counts.get(category, 0) + 1
                matched = True
                break
        if not matched: category_counts["Other/Unknown"] = category_counts.get("Other/Unknown", 0) + 1


    print("\nApproximate Business Category Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_queries_generated_valid * 100 if total_queries_generated_valid else 0
        print(f"  {category}: {count} queries ({percentage:.1f}%)")


if __name__ == "__main__":
    main()