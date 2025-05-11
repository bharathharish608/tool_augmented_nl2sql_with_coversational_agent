import random
from typing import Dict, List, Any

class ContrastivePairGenerator:
    def __init__(self):
        pass

    def generate_mismatched_pair(self, nl_sql_pair: Dict[str, Any], other_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a negative pair by mismatching NL and SQL from different pairs.
        """
        if not other_pairs:
            return {}
        other = random.choice(other_pairs)
        return {
            "nl": nl_sql_pair["nl"],
            "sql": other["sql"],
            "label": 0  # 0 for negative, 1 for positive
        }

    def generate_perturbed_sql_pair(self, nl_sql_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a negative pair by perturbing the SQL (e.g., changing a column name).
        Handles SELECT DISTINCT and avoids invalid SQL.
        """
        sql = nl_sql_pair["sql"]
        perturbed_sql = None
        if sql.strip().upper().startswith("SELECT DISTINCT "):
            perturbed_sql = sql.replace("SELECT DISTINCT ", "SELECT DISTINCT dummy_col, ", 1)
        elif sql.strip().upper().startswith("SELECT "):
            perturbed_sql = sql.replace("SELECT ", "SELECT dummy_col, ", 1)
        else:
            # If cannot safely perturb, return empty dict
            return {}
        return {
            "nl": nl_sql_pair["nl"],
            "sql": perturbed_sql,
            "label": 0
        }

if __name__ == "__main__":
    generator = ContrastivePairGenerator()
    pos_pair = {"nl": "How many items were sold last month?", "sql": "SELECT COUNT(*) FROM store_sales WHERE ..."}
    other_pairs = [
        {"nl": "What was the total revenue last year?", "sql": "SELECT SUM(amount) FROM sales WHERE ..."},
        {"nl": "List all customers from California.", "sql": "SELECT * FROM customers WHERE state = 'CA'"}
    ]
    neg_mismatch = generator.generate_mismatched_pair(pos_pair, other_pairs)
    neg_perturbed = generator.generate_perturbed_sql_pair(pos_pair)
    print("Positive pair:", pos_pair)
    print("Negative (mismatched):", neg_mismatch)
    print("Negative (perturbed SQL):", neg_perturbed) 