import anthropic
import os

class LLMInterface:
    def generate(self, prompt: str):
        raise NotImplementedError

class ClaudeLLMClient(LLMInterface):
    def __init__(self, api_key=None, model="claude-3-7-sonnet-20250219", max_tokens=1024):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content
        except Exception as e:
            print(f"ERROR: Claude API call failed: {e}")
            return None 