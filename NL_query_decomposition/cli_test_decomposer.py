import argparse
import json
from decomposer import ClaudeDecomposer

def main():
    parser = argparse.ArgumentParser(description="Schema-agnostic NL query decomposition using ClaudeDecomposer.")
    parser.add_argument('--query', '-q', type=str, help='Natural language query to decompose. If not provided, will prompt for input.')
    args = parser.parse_args()

    if args.query:
        nl_query = args.query
    else:
        nl_query = input("Enter a natural language query: ")

    decomposer = ClaudeDecomposer()
    ir = decomposer.decompose(nl_query)
    print(json.dumps(ir, indent=2))

if __name__ == "__main__":
    main()