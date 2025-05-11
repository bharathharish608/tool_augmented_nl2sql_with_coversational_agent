import csv

def extract_context_from_labelled_csv(labelled_csv_path, nl_query):
    """
    Given a labelled CSV and an NL query, return the relevant tables/columns as context nodes if present.
    """
    with open(labelled_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('nl', '').strip() == nl_query.strip():
                tables = [t.strip() for t in row.get('tables', '').split(';') if t.strip()]
                columns = [c.strip() for c in row.get('columns', '').split(';') if c.strip()]
                # Return fully qualified column names
                fq_columns = [f"{t}.{c}" for t in tables for c in columns if t and c]
                return tables + fq_columns
    return [] 