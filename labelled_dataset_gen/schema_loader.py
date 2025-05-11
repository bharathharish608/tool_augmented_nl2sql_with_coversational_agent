"""
Schema Loader for TPCDS Dataset Generation

This module loads and parses the TPCDS schema from schema.json,
providing structured access to tables, columns, and their metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from dataclasses import dataclass
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "global" / "config.env")

@dataclass
class Column:
    """Represents a column in the TPCDS schema with essential metadata."""
    name: str
    table: str
    data_type: str
    description: str
    is_key: bool = False
    is_foreign_key: bool = False
    # Statistical properties
    cardinality: Optional[float] = None
    uniqueness: Optional[float] = None
    min_value: Optional[Union[float, str]] = None
    max_value: Optional[Union[float, str]] = None
    total_count: Optional[float] = None
    null_count: Optional[float] = None
    # Foreign key relationships
    joins_with: List[str] = None
    
    def __post_init__(self):
        self.joins_with = self.joins_with or []
    
    @property
    def full_name(self) -> str:
        """Returns the fully qualified column name (table.column)."""
        return f"{self.table}.{self.name}"
    
    @property
    def null_percentage(self) -> Optional[float]:
        """Returns the percentage of null values if counts exist."""
        if self.total_count is not None and self.null_count is not None and self.total_count > 0:
            return (self.null_count / self.total_count) * 100
        return None

@dataclass
class Table:
    """Represents a table in the TPCDS schema."""
    name: str
    description: str
    columns: Dict[str, Column]
    key_columns: List[str]
    foreign_key_columns: List[str]
    column_count: int

    @property
    def primary_keys(self) -> List[str]:
        """Returns list of primary key column names."""
        return [col for col in self.columns if self.columns[col].is_key]
    
    @property
    def foreign_keys(self) -> List[str]:
        """Returns list of foreign key column names."""
        return [col for col in self.columns if self.columns[col].is_foreign_key]
    
    def get_column_stats(self) -> Dict[str, Dict[str, Any]]:
        """Returns statistics for all columns in the table."""
        stats = {}
        for col_name, col in self.columns.items():
            stats[col_name] = {
                "data_type": col.data_type,
                "null_percentage": col.null_percentage,
                "uniqueness": col.uniqueness,
                "cardinality": col.cardinality,
                "total_count": col.total_count,
                "min_value": col.min_value,
                "max_value": col.max_value
            }
        return stats

class SchemaLoader:
    """Loads and provides access to the TPCDS schema."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the schema loader.
        
        Args:
            schema_path: Path to the schema JSON file. If None, uses the path from config.
        """
        if schema_path is None:
            schema_path = os.getenv("SCHEMA_FILE", "tpcds_with_all_descriptions.json")
            schema_path = Path(__file__).parent.parent / "global" / schema_path
            
        self.schema_path = Path(schema_path)
        self.tables: Dict[str, Table] = {}
        self._load_schema()
        
    def _load_schema(self) -> None:
        """Load the schema from the JSON file."""
        try:
            with open(self.schema_path) as f:
                schema_data = json.load(f)
            
            # Parse tables
            for table_name, table_data in schema_data["tables"].items():
                columns = {}
                key_columns = table_data.get("key_columns", [])
                foreign_key_columns = table_data.get("foreign_key_columns", [])
                # New: columns is a dict, not a list
                for col_name, col_data in table_data["columns"].items():
                    columns[col_name] = Column(
                        name=col_data["name"],
                        table=table_name,
                        data_type=col_data.get("type", "unknown"),
                        description=col_data.get("description", ""),
                        cardinality=col_data.get("cardinality"),
                        uniqueness=col_data.get("uniqueness"),
                        min_value=col_data.get("min_value"),
                        max_value=col_data.get("max_value"),
                        total_count=col_data.get("total_count"),
                        null_count=col_data.get("null_count"),
                        joins_with=col_data.get("joins_with", []),
                        is_key=col_name in key_columns,
                        is_foreign_key=col_name in foreign_key_columns
                    )
                self.tables[table_name] = Table(
                    name=table_name,
                    description=table_data.get("description", ""),
                    columns=columns,
                    key_columns=key_columns,
                    foreign_key_columns=foreign_key_columns,
                    column_count=len(columns)
                )
            logger.info(f"Successfully loaded schema with {len(self.tables)} tables")
        except FileNotFoundError:
            logger.error(f"Schema file not found: {self.schema_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in schema file: {self.schema_path}")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in schema: {e}")
            raise
    
    def get_table(self, table_name: str) -> Optional[Table]:
        """Get a table by name."""
        return self.tables.get(table_name)
    
    def get_column(self, table_name: str, column_name: str) -> Optional[Column]:
        """Get a column by table and column name."""
        table = self.get_table(table_name)
        if table:
            return table.columns.get(column_name)
        return None
    
    def get_all_table_names(self) -> List[str]:
        """Get list of all table names."""
        return list(self.tables.keys())
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """Get list of column names for a table."""
        table = self.get_table(table_name)
        return list(table.columns.keys()) if table else []
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """Get list of primary key columns for a table."""
        table = self.get_table(table_name)
        return table.primary_keys if table else []
    
    def get_foreign_keys(self, table_name: str) -> List[str]:
        """Get list of foreign key columns for a table."""
        table = self.get_table(table_name)
        return table.foreign_keys if table else []

if __name__ == "__main__":
    # Initialize schema loader
    schema_loader = SchemaLoader()
    
    # Print schema information
    print(f"Loaded {len(schema_loader.get_all_table_names())} tables:")
    
    # Show summary of all tables
    for table_name in schema_loader.get_all_table_names():
        table = schema_loader.get_table(table_name)
        print(f"\n{table_name}:")
        print(f"Description: {table.description[:100]}...")
        print(f"Columns: {table.column_count}")
        print(f"Primary Keys: {schema_loader.get_primary_keys(table_name)}")
    
    # Show detailed metadata for a specific column
    print("\n=== Example Column Metadata ===")
    # Pick a table/column that exists in the new schema
    example_table = next(iter(schema_loader.tables.keys()))
    example_col = next(iter(schema_loader.tables[example_table].columns.keys()))
    col = schema_loader.get_column(example_table, example_col)
    if col:
        print(f"\nColumn: {col.full_name}")
        print(f"Data Type: {col.data_type}")
        print(f"Description: {col.description}")
        print(f"Total Count: {col.total_count}")
        print(f"Null Count: {col.null_count}")
        print(f"Null %: {f'{col.null_percentage:.2f}%' if col.null_percentage is not None else 'N/A'}")
        print(f"Cardinality: {col.cardinality}")
        print(f"Uniqueness: {col.uniqueness}")
        print(f"Range: {col.min_value} to {col.max_value}")
        print(f"Joins with: {', '.join(col.joins_with)}") 