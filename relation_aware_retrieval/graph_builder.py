import json
import csv
import networkx as nx
from pathlib import Path
import re

class SchemaGraphBuilder:
    def __init__(self, schema_json_path, labelled_csv_path):
        self.schema_json_path = schema_json_path
        self.labelled_csv_path = labelled_csv_path
        self.graph = nx.Graph()
        self.schema = None
        self._load_schema()
        self._add_schema_nodes_and_edges()
        self._infer_relationships_from_labelled_sql()

    def _load_schema(self):
        with open(self.schema_json_path) as f:
            self.schema = json.load(f)

    def _add_schema_nodes_and_edges(self):
        for table, tdata in self.schema['tables'].items():
            self.graph.add_node(table, type='table', description=tdata.get('description', ''))
            columns = tdata.get('columns', {})
            for col in columns:
                col_name = f"{table}.{col}"
                self.graph.add_node(col_name, type='column', description=columns[col].get('description', '') if isinstance(columns[col], dict) else columns[col])
                self.graph.add_edge(table, col_name, type='contains', weight=1.0)
            # Add explicit foreign keys if present
            for fk in tdata.get('foreign_keys', []):
                src = f"{table}.{fk['column']}"
                tgt = fk['references']  # should be fully qualified
                self.graph.add_edge(src, tgt, type='foreign_key', weight=1.0)
                tgt_table = tgt.split('.')[0]
                self.graph.add_edge(table, tgt_table, type='table_fk', weight=1.0)

    def _infer_relationships_from_labelled_sql(self):
        # Parse the labelled CSV and extract join relationships from SQL
        with open(self.labelled_csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sql = row.get('sql', '')
                if sql:
                    self._extract_joins_from_sql(sql)

    def _extract_joins_from_sql(self, sql):
        # Simple regex-based join extraction (can be improved)
        join_pattern = re.compile(r'JOIN\s+([\w\.]+)\s+ON\s+([\w\.]+)\s*=\s*([\w\.]+)', re.IGNORECASE)
        for match in join_pattern.finditer(sql):
            table2, col1, col2 = match.groups()
            # Add nodes if not present
            for node in [col1, col2, table2]:
                if node not in self.graph:
                    self.graph.add_node(node, type='unknown')
            # Add join edge
            self.graph.add_edge(col1, col2, type='sql_inferred_join', weight=1.0)
            # Optionally, add table-to-table edge
            table1 = col1.split('.')[0]
            if table1 != table2:
                self.graph.add_edge(table1, table2, type='sql_inferred_table_join', weight=1.0)

    def get_graph(self):
        return self.graph 