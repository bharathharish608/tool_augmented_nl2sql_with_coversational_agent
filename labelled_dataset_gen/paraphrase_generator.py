from typing import List
from llm_interface import LLMInterface

class ParaphraseGenerator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def generate_paraphrases(self, nl_question: str, n: int = 3) -> List[str]:
        """
        Generate n paraphrases for a given NL question using the LLM interface.
        """
        prompt = (
            f"Paraphrase the following question in {n} different ways. "
            "Each paraphrase should preserve the original meaning but use different wording.\n"
            f"Question: {nl_question}"
        )
        # The LLM is expected to return a list of paraphrases or a string with n lines
        response = self.llm.generate(prompt)
        if isinstance(response, list):
            return response
        elif isinstance(response, str):
            # Split by lines and clean up
            return [line.strip('- ').strip() for line in response.strip().split('\n') if line.strip()]
        else:
            return []

if __name__ == "__main__":
    # Minimal usage example with a mock LLMInterface
    class MockLLM(LLMInterface):
        def generate(self, prompt: str):
            return [
                "What is the total sales for last month?",
                "How much did we sell last month?",
                "Can you tell me the sales figures for last month?"
            ]
    llm = MockLLM()
    paraphraser = ParaphraseGenerator(llm)
    question = "How many items were sold last month?"
    paraphrases = paraphraser.generate_paraphrases(question, n=3)
    print("Original:", question)
    print("Paraphrases:")
    for p in paraphrases:
        print("-", p) 