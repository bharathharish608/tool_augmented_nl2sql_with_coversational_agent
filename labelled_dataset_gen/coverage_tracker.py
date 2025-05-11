from typing import Set, Dict, List, Optional
from schema_loader import SchemaLoader

class CoverageTracker:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader
        self.all_tables = set(schema_loader.get_all_table_names())
        self.all_columns = set(
            (t, c)
            for t in self.all_tables
            for c in schema_loader.get_table_columns(t)
        )
        self.covered_tables: Set[str] = set()
        self.covered_columns: Set[tuple] = set()

    def update(self, tables: List[str], columns: List[str]):
        self.covered_tables.update(tables)
        # columns are expected as table.column or (table, column)
        for col in columns:
            if isinstance(col, str) and "." in col:
                t, c = col.split(".", 1)
                self.covered_columns.add((t, c))
            elif isinstance(col, tuple) and len(col) == 2:
                self.covered_columns.add(col)

    def table_coverage(self) -> float:
        return len(self.covered_tables) / len(self.all_tables) * 100 if self.all_tables else 0.0

    def column_coverage(self) -> float:
        return len(self.covered_columns) / len(self.all_columns) * 100 if self.all_columns else 0.0

    def uncovered_tables(self) -> Set[str]:
        return self.all_tables - self.covered_tables

    def uncovered_columns(self) -> Set[tuple]:
        return self.all_columns - self.covered_columns

    def suggest_tables(self, n: int = 1) -> List[str]:
        # Suggest n random uncovered tables
        import random
        candidates = list(self.uncovered_tables())
        return random.sample(candidates, min(n, len(candidates))) if candidates else []

    def suggest_columns(self, n: int = 1) -> List[tuple]:
        # Suggest n random uncovered columns
        import random
        candidates = list(self.uncovered_columns())
        return random.sample(candidates, min(n, len(candidates))) if candidates else []

if __name__ == "__main__":
    loader = SchemaLoader()
    tracker = CoverageTracker(loader)
    print(f"Initial table coverage: {tracker.table_coverage():.2f}%")
    print(f"Initial column coverage: {tracker.column_coverage():.2f}%")
    # Simulate an update
    tracker.update(["web_returns"], ["web_returns.wr_item_sk", ("web_returns", "wr_returned_date_sk")])
    print(f"After update, table coverage: {tracker.table_coverage():.2f}%")
    print(f"After update, column coverage: {tracker.column_coverage():.2f}%")
    print(f"Uncovered tables: {tracker.uncovered_tables()}")
    print(f"Uncovered columns (sample): {list(tracker.uncovered_columns())[:5]}")
    print(f"Suggest tables: {tracker.suggest_tables(2)}")
    print(f"Suggest columns: {tracker.suggest_columns(2)}") 