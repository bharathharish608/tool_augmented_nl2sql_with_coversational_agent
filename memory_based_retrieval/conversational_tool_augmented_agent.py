from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from memory_based_retrieval.langchain_tools import relation_aware_tool, all_tools
import os
from dotenv import load_dotenv

# Load Claude API key from environment or config.env
load_dotenv("global/config.env")

llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=8192
)

# Short-term (session) memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [relation_aware_tool]

SYSTEM_PROMPT = """
IMPORTANT: You must always respond using ONLY the following format:
Thought: <your reasoning>
Action: <one of [ListTables, GetColumns, SearchSchema, SemanticSearchSchema, RelationAwareRetrieval, GenerateIR, GenerateSQLPseudocode, GenerateSQL]>
Action Input: <input>
Do NOT output SQL, IR, or final answers unless you are explicitly instructed to do so by the tool invocation. Do NOT include explanations, markdown, or any extra text outside the format above. If you are unsure, ask for clarification.
"""

agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=False,
    max_iterations=20
)

def main():
    print("Conversational Tool-Augmented Agent (Claude + Memory)")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        try:
            response = agent.invoke({"input": user_input}, return_intermediate_steps=True)
            print("\n--- Agent Output ---")
            print(response.get("output", response))
            print("\n--- Intermediate Steps ---")
            for step in response.get("intermediate_steps", []):
                print(step)
        except Exception as e:
            print(f"[ERROR] {e}")
            print("Raw LLM output (if available):")
            try:
                print(response)
            except:
                pass

if __name__ == "__main__":
    main() 