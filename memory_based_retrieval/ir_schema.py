from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class IRModel(BaseModel):
    """
    Pydantic model for the NL2SQL Intermediate Representation (IR).
    """
    query_understanding: str
    entities: Dict[str, Any]
    metrics: Optional[List[Any]] = None
    filters: Optional[List[Any]] = None
    time_dimensions: Optional[List[Any]] = None
    groupings: Optional[List[Any]] = None
    execution_plan: Optional[List[str]] = None
    complete_sql_pseudocode: Optional[str] = None 