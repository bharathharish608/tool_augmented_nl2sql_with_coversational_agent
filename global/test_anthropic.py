import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load config.env
load_dotenv(Path(__file__).parent / "config.env")

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-3-7-sonnet-20250219"

if not API_KEY:
    print("ANTHROPIC_API_KEY not found in environment. Please check config.env.")
    exit(1)

client = anthropic.Anthropic(api_key=API_KEY)

try:
    print("Sending test prompt to Anthropic Claude...")
    message = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "Hello! Please reply with 'pong' if you can hear me."}]
    )
    print("\n--- Anthropic API Response ---\n")
    print(message)
    print("\n--- End Response ---\n")
except Exception as e:
    print(f"Error communicating with Anthropic API: {e}") 