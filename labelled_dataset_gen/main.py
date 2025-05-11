import random
import json
import re
from schema_loader import SchemaLoader
from coverage_tracker import CoverageTracker
from prompt_generator import PromptGenerator
from llm_interface import ClaudeLLMClient
from paraphrase_generator import ParaphraseGenerator
from contrastive_pair_generator import ContrastivePairGenerator
from validator import Validator
from metadata_annotator import MetadataAnnotator
from dataset_writer import DatasetWriter

BUSINESS_INTENTS = [
    "Sales analysis", "Customer segmentation", "Inventory management", "Promotion effectiveness",
    "Revenue trends", "Product performance", "Geographic analysis", "Time-based analysis"
]
COMPLEXITIES = ["easy", "medium", "hard"]

MAX_EXAMPLES = 1000
BATCH_SIZE = 5

def extract_json_from_text(text):
    # Remove markdown code block if present
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Try to parse JSON
    try:
        return json.loads(text)
    except Exception:
        return None

def robust_llm_output_to_dict(result):
    # Handle custom object with 'text' attribute
    if hasattr(result, 'text'):
        result = result.text
    # If result is a string, try to extract JSON
    if isinstance(result, str):
        parsed = extract_json_from_text(result)
        if not parsed:
            print("Warning: Could not extract JSON from LLM string output, skipping.")
            return None
        result = parsed
    # If result is a list, try to use the first element
    if isinstance(result, list):
        if not result:
            print("Warning: LLM output is an empty list, skipping.")
            return None
        first = result[0]
        # If first element is a dict, use it
        if isinstance(first, dict):
            result = first
        # If first element is a custom object with 'text', extract and parse
        elif hasattr(first, 'text'):
            parsed = extract_json_from_text(first.text)
            if not parsed:
                print("Warning: Could not extract JSON from first element's text, skipping.")
                return None
            result = parsed
        # If first element is a string, extract JSON
        elif isinstance(first, str):
            parsed = extract_json_from_text(first)
            if not parsed:
                print("Warning: Could not extract JSON from first element string, skipping.")
                return None
            result = parsed
        else:
            print("Warning: LLM output list's first element is not usable, skipping.")
            return None
    # Final check
    if not isinstance(result, dict):
        print("Warning: LLM output is not a dict after all conversions, skipping:", result)
        return None
    return result

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, 'text'):
        return obj.text
    else:
        return obj

if __name__ == "__main__":
    schema_loader = SchemaLoader()
    coverage_tracker = CoverageTracker(schema_loader)
    prompt_gen = PromptGenerator(schema_loader)
    llm = ClaudeLLMClient()  # Replace with your LLM interface as needed
    paraphraser = ParaphraseGenerator(llm)
    contrastive_gen = ContrastivePairGenerator()
    validator = Validator(schema_loader)
    annotator = MetadataAnnotator(schema_loader)
    examples = []
    last_written = 0

    while len(examples) < MAX_EXAMPLES:
        # 1. Select tables for prompt (rotate for coverage)
        tables = coverage_tracker.suggest_tables(n=2)
        if not tables:
            tables = random.sample(schema_loader.get_all_table_names(), 2)
        business_intent = random.choice(BUSINESS_INTENTS)
        complexity = random.choice(COMPLEXITIES)
        prompt = prompt_gen.generate_prompt(
            table_names=tables,
            business_intent=business_intent,
            complexity=complexity
        )
        # 2. Generate NL/SQL pair from LLM
        result = llm.generate(prompt)
        result = robust_llm_output_to_dict(result)
        if not result:
            continue
        # 3. Generate paraphrases
        result['paraphrases'] = paraphraser.generate_paraphrases(result['nl'])
        # 4. Validate
        if not validator.validate_example(result):
            continue
        # 5. Annotate metadata
        annotated = annotator.annotate(result)
        # 6. Update coverage
        coverage_tracker.update(annotated['tables'], annotated['columns'])
        # 7. Add positive example
        annotated['label'] = 1
        examples.append(annotated)
        # 8. Generate contrastive (negative) pairs
        # Mismatched pair (if more than 1 example exists)
        if len(examples) > 1:
            neg_mismatch = contrastive_gen.generate_mismatched_pair(annotated, examples[:-1])
            if neg_mismatch and 'nl' in neg_mismatch and 'sql' in neg_mismatch:
                try:
                    neg_mismatch = annotator.annotate(neg_mismatch)
                    neg_mismatch['label'] = 0
                    examples.append(neg_mismatch)
                except Exception as e:
                    print(f"Skipping mismatched negative pair due to error: {e}")
        # Perturbed SQL pair
        neg_perturbed = contrastive_gen.generate_perturbed_sql_pair(annotated)
        if neg_perturbed and 'nl' in neg_perturbed and 'sql' in neg_perturbed:
            try:
                neg_perturbed = annotator.annotate(neg_perturbed)
                neg_perturbed['label'] = 0
                examples.append(neg_perturbed)
            except Exception as e:
                print(f"Skipping perturbed negative pair due to error: {e}")
        # 9. Log progress and write batch
        if len(examples) % 100 == 0 and len(examples) > last_written:
            print(f"Generated {len(examples)} examples... Writing batch to disk.")
            # Ensure all examples are JSON serializable
            batch = [make_json_serializable(ex) for ex in examples]
            batch = DatasetWriter.deduplicate(batch)
            fieldnames = ["nl", "paraphrases", "sql", "tables", "columns", "business_intent", "complexity", "alternatives", "label"]
            DatasetWriter.write_csv(batch, "labelled_dataset_gen/datasets/labelled_dataset.csv", fieldnames)
            DatasetWriter.write_jsonl(batch, "labelled_dataset_gen/datasets/labelled_dataset.jsonl")
            last_written = len(examples)
    # Final write after loop
    examples = [make_json_serializable(ex) for ex in examples]
    examples = DatasetWriter.deduplicate(examples)
    fieldnames = ["nl", "paraphrases", "sql", "tables", "columns", "business_intent", "complexity", "alternatives", "label"]
    DatasetWriter.write_csv(examples, "labelled_dataset_gen/datasets/labelled_dataset.csv", fieldnames)
    DatasetWriter.write_jsonl(examples, "labelled_dataset_gen/datasets/labelled_dataset.jsonl")
    print(f"Done! Wrote {len(examples)} examples to labelled_dataset_gen/datasets/labelled_dataset.csv and .jsonl") 