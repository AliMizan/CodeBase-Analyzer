from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AnalysisConfig(BaseModel):
    max_files: int = 50
    enable_security_scan: bool = True
    enable_complexity_analysis: bool = True
    enable_duplicate_detection: bool = True
    enable_dependency_analysis: bool = True
    enable_test_analysis: bool = True
    confidence_threshold: float = 0.6
    parallel_workers: int = 4
    enable_semantic_analysis: bool = True
    semantic_collection_name: str = "codebase_analysis"
    semantic_persist_dir: str = "./chroma_db"

class AnalyzeRequest(BaseModel):
    directory: str
    project_name: Optional[str]
    config: Optional[AnalysisConfig]

class QARequest(BaseModel):
    question: str

class SearchResponse(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str

class Issue(BaseModel):
    file_path: str
    line_number: int
    category: str
    severity: str
    description: str
    suggestion: str
    confidence: float
    issue_type: Optional[str] = "general"   # or default

    

class ReportResponse(BaseModel):
    timestamp: str
    project_name: str
    summary: str
    total_files_analyzed: int
    total_lines_of_code: int
    total_issues_found: int
    security_score: float
    maintainability_score: float
    issues: List[Issue]

class GitCloneRequest(BaseModel):
    github_url: str
    project_name: Optional[str]
    config: Optional[AnalysisConfig]
