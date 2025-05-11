from .graph_builder import SchemaGraphBuilder
from .context_extraction import extract_context_from_labelled_csv
from .rwr import context_aware_rwr

class RelationAwareRetriever:
    def __init__(self, schema_json_path, labelled_csv_path):
        self.graph_builder = SchemaGraphBuilder(schema_json_path, labelled_csv_path)
        self.graph = self.graph_builder.get_graph()
        self.labelled_csv_path = labelled_csv_path

    def retrieve(self, nl_query, top_k=10):
        context_nodes = extract_context_from_labelled_csv(self.labelled_csv_path, nl_query)
        if not context_nodes:
            # fallback: use all table nodes as context
            context_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'table']
        scores = context_aware_rwr(self.graph, context_nodes)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k] 