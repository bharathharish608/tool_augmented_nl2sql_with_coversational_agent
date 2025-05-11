from graph_builder import SchemaGraphBuilder
from pathlib import Path
from collections import Counter

if __name__ == "__main__":
    schema_json = Path("global/tpcds_with_all_descriptions.json")
    labelled_csv = Path("labelled_dataset_gen/datasets/labelled_dataset.csv")
    builder = SchemaGraphBuilder(schema_json, labelled_csv)
    G = builder.get_graph()
    print(f"Graph built! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print("Sample nodes:", list(G.nodes)[:10])
    print("Sample edges:", list(G.edges(data=True))[:10])

    # Print sample of inferred join edges
    inferred_joins = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('type') == 'sql_inferred_join']
    print(f"\nSample inferred join edges (up to 10): {inferred_joins[:10]}")

    # Table occurrence frequencies in inferred joins
    table_counter = Counter()
    for u, v, d in inferred_joins:
        for node in (u, v):
            if '.' in node:
                table = node.split('.')[0]
                table_counter[table] += 1
    print(f"\nTable occurrence frequencies in inferred joins (top 10): {table_counter.most_common(10)}")

    # Print edge weights for a sample of edges
    print("\nSample edge weights:")
    for u, v, d in list(G.edges(data=True))[:10]:
        print(f"Edge {u} -- {v}: weight={d.get('weight', 1.0)}, type={d.get('type')}") 