# Memory-Based Retrieval for NL2SQL

## Purpose
This module introduces memory-augmented retrieval to NL2SQL systems, enabling the agent to reason more efficiently and contextually over large, complex schemas. By leveraging short-term, long-term, and semantic memory, the agent can:
- Avoid redundant schema exploration
- Support multi-turn and follow-up queries
- Disambiguate ambiguous terms using learned relationships
- Personalize and adapt to user or organizational patterns

## Memory Types
- **Short-term (Session) Memory:** Remembers the current session's context, tool calls, clarifications, and explored schema elements.
- **Long-term (Agentic) Memory:** Persists across sessions, storing common join paths, user preferences, and schema evolution.
- **Semantic/Relational Memory:** Captures synonym mappings, reasoning chains, and multi-hop join paths for robust, context-aware retrieval.

## Benefits
- **Efficiency:** Reduces repeated schema scans and clarifications.
- **Contextual Reasoning:** Supports follow-up and multi-turn queries naturally.
- **Smarter Retrieval:** Leverages semantic and relational knowledge for better disambiguation and join path selection.
- **Personalization:** Learns from user behavior and query patterns over time.

## Architecture
See [arch.md](./arch.md) for a detailed architecture diagram and memory flow.

## Example Use Cases
- User asks a follow-up query; agent reuses previous context.
- Ambiguous terms are resolved using semantic memory.
- Agent adapts to recurring query patterns or user preferences.

## Getting Started
- Review the architecture in `docs/arch.md`.
- Integrate memory modules into your agent pipeline for enhanced NL2SQL reasoning.

---

*This module is inspired by advanced agentic frameworks (e.g., LangChain) and is designed for extensibility and real-world scale.* 