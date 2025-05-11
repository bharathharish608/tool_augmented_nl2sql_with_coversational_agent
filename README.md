# NL2SQL Full Pipeline

## Overview
A modular, production-grade natural language to SQL (NL2SQL) pipeline that translates analytical questions into SQL for any relational schema. Features robust schema exploration, semantic search, graph-based relation-aware retrieval, memory-augmented conversational agent, intermediate representation (IR), SQL pseudocode, and dialect-specific SQL generation. Not tied to TPCDS—works with any schema.

## Key Features
- **Schema-Agnostic:** Works with any SQL schema (star, snowflake, normalized, etc.)
- **Tool-Augmented Agent:** LLM agent uses schema tools for exploration and reasoning
- **Semantic Search:** BM25+SBERT fusion for robust synonym and ambiguity resolution
- **Relation-Aware Retrieval:** Graph-based, context-aware retrieval using Random Walk with Restart (RWR) seeded by LLM context
- **Memory-Augmented Agent:** Conversational agent with short-term and long-term memory (LangChain + Claude)
- **Anthropic Function Calling:** Uses Claude 3.7 tool use for robust, structured IR generation
- **IR/Pseudocode/SQL Separation:** Three-step LLM pipeline for reliability and debuggability
- **Clarification & Circuit Breakers:** Asks for clarification if ambiguous or stuck
- **Configurable & Extensible:** Easily onboard new schemas and models
- **Structured Output Enforcement:** Pydantic and LangChain output parsers for IR
- **SQL Validation:** SQLGlot validation and dialect support

## Folder Structure
```
global/
  config.env           # API keys and config
  tpcds_with_all_descriptions.json  # Example schema (replaceable)
  requirements.txt     # Python dependencies
  docs/                # Documentation (this folder)
NL_query_decomposition/
  tool_augmented_agent.py  # Main agent entry point
  sql_generator.py         # SQL generation from IR/pseudocode
  schema_tools.py          # Schema loading and tool APIs
  ...
semantic_schema_search/    # BM25, SBERT, fusion, training, eval
relation_aware_retrieval/  # Graph-based retrieval modules
memory_based_retrieval/
  langchain_tools.py       # All tools for LangChain agent
  conversational_tool_augmented_agent.py  # Memory-augmented agent entry point
  ir_function_calling_demo.py             # Anthropic function-calling IR demo
```

## How to Use
1. **Install dependencies:**
   ```bash
   pip install -r global/requirements.txt
   ```
2. **Configure API keys:**
   - Edit `global/config.env` with your Anthropic API key and any other settings.
3. **Load your schema:**
   - Place your schema JSON in `global/` and update `schema_tools.py` to load it.
4. **Run the memory-augmented agent:**
   ```bash
   python3 -m memory_based_retrieval.conversational_tool_augmented_agent
   ```
   - The agent will:
     - Explore the schema using all tools (list, search, relation-aware, etc.)
     - Generate an IR (via Claude function calling)
     - Generate SQL pseudocode
     - Generate and output SQL (raw LLM output, no forced parsing)
     - Maintain conversational memory
     - Output all intermediate steps and tool calls

## Entry Points
- **Conversational agent:** `memory_based_retrieval/conversational_tool_augmented_agent.py`
- **All LangChain tools:** `memory_based_retrieval/langchain_tools.py`
- **IR function-calling demo:** `memory_based_retrieval/ir_function_calling_demo.py`
- **Classic agent:** `NL_query_decomposition/tool_augmented_agent.py`
- **SQL generator:** `NL_query_decomposition/sql_generator.py`
- **Schema tools:** `NL_query_decomposition/schema_tools.py`

## Configuration
- **API keys:** Set in `global/config.env`
- **Schema:** Update `tpcds_with_all_descriptions.json` or add your own
- **Model selection:** Change model in agent or SQL generator as needed

## Extending to New Schemas
- Add your schema JSON to `global/`
- Update `schema_tools.py` to load and expose the new schema
- No other code changes needed—agent is schema-agnostic

## Sample Questions
- "List all customers from Canada"
- "For each region, show top 5 products by sales in 2023"
- "What is the average shipping time by mode and warehouse location?"
- "Show total revenue by customer state for Electronics, Clothing, and Home categories over the last 12 months"
- "For each customer demographic segment, what is the average shipping time and net profit for orders that were returned, grouped by shipping mode and warehouse location, in the last year?"

## Sample Multi-Turn Conversation
```
User: What is the total quantity sold and revenue by category and subcategory over the last 12 months?
Agent: [Explores schema, generates IR, outputs SQL]

User: For categories 'Electronics', 'Clothing', and 'Home', what is the total revenue by customer state over the last 12 months?
Agent: [Uses memory, schema search, outputs SQL]

User: For each customer demographic segment, what is the average shipping time and net profit for orders that were returned, grouped by shipping mode and warehouse location, in the last year?
Agent: [Uses relation-aware retrieval, outputs SQL]
```

## Troubleshooting & Tips
- **LLM output truncated?** Increase `max_tokens` in agent and SQL generator
- **Ambiguous queries?** Agent will ask for clarification
- **New schema?** Just update schema file and reload
- **Logs:** Check console/logs for stepwise progress and errors
- **Output parsing errors:** The agent now prints raw LLM output if parsing fails, so you always see the answer

---

For advanced usage, see code comments and modular design in each file. 