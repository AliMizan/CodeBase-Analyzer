

# import json
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# from .schemas import ReportResponse, Issue
# from new2 import EnhancedCodebaseAnalyzer, AnalysisConfig

# REPORTS_DIR = Path("reports")
# REPORTS_DIR.mkdir(exist_ok=True)

# class AnalyzerService:
#     def __init__(self):
#         self.analyzers: Dict[str, EnhancedCodebaseAnalyzer] = {}

#     def _get_analyzer(self, project_name: str) -> EnhancedCodebaseAnalyzer:
#         if project_name not in self.analyzers:
#             raise ValueError("Analysis not found for project")
#         return self.analyzers[project_name]

#     def analyze(self, directory: str, project_name: str, config_data: Optional[Dict[str, Any]]) -> ReportResponse:
#         config = AnalysisConfig(**config_data) if config_data else AnalysisConfig()
#         analyzer = EnhancedCodebaseAnalyzer(config=config)

#         # Run analysis
#         report = analyzer.analyze_codebase(directory=directory, project_name=project_name)

#         # Save analyzer in memory (for QA, search, etc.)
#         self.analyzers[project_name] = analyzer

#         # Save report as JSON file
#         report_path = REPORTS_DIR / f"{project_name}.json"
#         with open(report_path, "w", encoding="utf-8") as f:
#             json.dump(report.dict(), f, indent=2, default=str)

#         return self._convert_report(report)

#     def get_report(self, project_name: str) -> ReportResponse:
#         report_path = REPORTS_DIR / f"{project_name}.json"
#         if not report_path.exists():
#             raise ValueError("Report not found. Run analysis first.")
#         with open(report_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         return self._convert_report_dict(data)

#     def get_issues(self, project_name: str, category: Optional[str], severity: Optional[str]) -> List[Issue]:
#         report = self.get_report(project_name)
#         issues = report.issues
#         if category:
#             issues = [i for i in issues if i.category == category]
#         if severity:
#             issues = [i for i in issues if i.severity == severity]
#         return issues

#     def ask_question(self, project_name: str, question: str) -> Dict[str, Any]:
#         analyzer = self._get_analyzer(project_name)
#         return analyzer.ask_codebase_question(question)

#     def search(self, project_name: str, query: str) -> List[Dict[str, Any]]:
#         analyzer = self._get_analyzer(project_name)
#         results = analyzer.search_codebase(query=query)
#         formatted = []
#         for r in results:
#             formatted.append({
#                 "file_path": r["metadata"]["file_path"],
#                 "start_line": r["metadata"]["start_line"],
#                 "end_line": r["metadata"]["end_line"],
#                 "content": r["content"]
#             })
#         return formatted

#     def get_trends(self, project_name: str) -> Dict[str, Any]:
#         # Trends only work if analyzer.db is used
#         # Here, we can just return dummy or JSON-based history
#         return {"message": "Trends not available when storing only JSON reports."}

#     def get_semantic_stats(self, project_name: str) -> Dict[str, Any]:
#         analyzer = self._get_analyzer(project_name)
#         return analyzer.get_semantic_stats()

#     def _convert_report(self, report) -> ReportResponse:
#         """Convert CodebaseReport object into FastAPI schema"""
#         return ReportResponse(
#             timestamp=report.timestamp,
#             project_name=report.project_name,
#             summary=report.summary,
#             total_files_analyzed=report.total_files_analyzed,
#             total_lines_of_code=report.total_lines_of_code,
#             total_issues_found=report.total_issues_found,
#             security_score=report.security_score,
#             maintainability_score=report.maintainability_score,
#             issues=[
#                 Issue(
#                     file_path=issue.file_path,
#                     line_number=issue.line_number,
#                     category=issue.category,
#                     severity=issue.severity,
#                     description=issue.description,
#                     suggestion=issue.suggestion,
#                     confidence=issue.confidence
#                 ) for issue in report.issues
#             ]
#         )

#     def _convert_report_dict(self, data: Dict[str, Any]) -> ReportResponse:
#         """Convert dict loaded from JSON into FastAPI schema"""
#         return ReportResponse(
#             timestamp=data["timestamp"],
#             project_name=data["project_name"],
#             summary=data["summary"],
#             total_files_analyzed=data["total_files_analyzed"],
#             total_lines_of_code=data["total_lines_of_code"],
#             total_issues_found=data["total_issues_found"],
#             security_score=data["security_score"],
#             maintainability_score=data["maintainability_score"],
#             issues=[
#                 Issue(
#                     file_path=issue["file_path"],
#                     line_number=issue["line_number"],
#                     category=issue["category"],
#                     severity=issue["severity"],
#                     description=issue["description"],
#                     suggestion=issue["suggestion"],
#                     confidence=issue.get("confidence", 0.8)
#                 ) for issue in data.get("issues", [])
#             ]
#         )

import json
from typing import List, Dict, Any, Optional
from .schemas import ReportResponse, Issue
from pathlib import Path
from analyzer import EnhancedCodebaseAnalyzer, AnalysisConfig
import subprocess
import tempfile

REPORT_DIR = Path("./reports")
REPORT_DIR.mkdir(exist_ok=True)

class AnalyzerService:
    def __init__(self):
        self.analyzers: Dict[str, EnhancedCodebaseAnalyzer] = {}

    def _get_report_path(self, project_name: str) -> Path:
        return REPORT_DIR / f"{project_name}.json"

    def _get_analyzer(self, project_name: str) -> EnhancedCodebaseAnalyzer:
        if project_name in self.analyzers:
            return self.analyzers[project_name]

        report_path = self._get_report_path(project_name)
        if not report_path.exists():
            raise ValueError("Analysis not found for project")

        # Load report data
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Create a new analyzer instance
        config = AnalysisConfig()  # Or you can store and load config if needed
        analyzer = EnhancedCodebaseAnalyzer(config=config)

        # Optionally, you can reload or reindex files here if necessary
        # For now, we just keep it alive in memory
        self.analyzers[project_name] = analyzer
        return analyzer

    def analyze(self, directory: str, project_name: str, config_data: Optional[Dict[str, Any]]) -> ReportResponse:

        semantic_db_path = f"./chroma_db/{project_name}"
        semantic_collection = f"{project_name}_collection"

        






        config = AnalysisConfig(**config_data) if config_data else AnalysisConfig()
        config.semantic_persist_dir = semantic_db_path
        config.semantic_collection_name = semantic_collection

        analyzer = EnhancedCodebaseAnalyzer(config=config)
        report = analyzer.analyze_codebase(directory=directory, project_name=project_name)

        # Store in memory
        self.analyzers[project_name] = analyzer

        # Save report to JSON file
        report_path = self._get_report_path(project_name)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.dict(), f, indent=2, default=str)

        return self._convert_report(report)
    
    def clone_and_analyze(self, github_url: str, project_name: str, config_data: Optional[Dict[str, Any]]) -> ReportResponse:
    # Clone into a temp dir (or named dir for project)
     target_dir = Path(f"./{project_name}")
     if target_dir.exists():
        # optionally delete or reuse
        subprocess.run(["rm", "-rf", str(target_dir)], check=True)

     subprocess.run(["git", "clone", github_url, str(target_dir)], check=True)

    # Run analysis on cloned dir
     return self.analyze(str(target_dir), project_name, config_data)

    
    

    def get_report(self, project_name: str) -> ReportResponse:
        report_path = self._get_report_path(project_name)
        if not report_path.exists():
            raise ValueError("Report file not found")

        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        return ReportResponse(**report_data)

    def get_issues(self, project_name: str, category: Optional[str], severity: Optional[str]) -> List[Issue]:
        report = self.get_report(project_name)
        issues = report.issues
        if category:
            issues = [i for i in issues if i.category == category]
        if severity:
            issues = [i for i in issues if i.severity == severity]
        return issues

    def ask_question(self, project_name: str, question: str) -> Dict[str, Any]:
        analyzer = self._get_analyzer(project_name)
        result = analyzer.ask_codebase_question(question)
        return result

    def search(self, project_name: str, query: str) -> List[Dict[str, Any]]:
        analyzer = self._get_analyzer(project_name)
        results = analyzer.search_codebase(query=query)
        formatted = []
        for r in results:
            formatted.append({
                "file_path": r["metadata"]["file_path"],
                "start_line": r["metadata"]["start_line"],
                "end_line": r["metadata"]["end_line"],
                "content": r["content"]
            })
        return formatted

    def get_trends(self, project_name: str) -> Dict[str, Any]:
        analyzer = self._get_analyzer(project_name)
        return analyzer.db_manager.get_trends(project_name)

    def get_semantic_stats(self, project_name: str) -> Dict[str, Any]:
        analyzer = self._get_analyzer(project_name)
        return analyzer.get_semantic_stats()
    
    def list_projects(self) -> List[str]:
     report_files = REPORT_DIR.glob("*.json")
     projects = [file.stem for file in report_files]
     return projects


    def _convert_report(self, report) -> ReportResponse:
        return ReportResponse(
            timestamp=report.timestamp,
            project_name=report.project_name,
            summary=report.summary,
            total_files_analyzed=report.total_files_analyzed,
            total_lines_of_code=report.total_lines_of_code,
            total_issues_found=report.total_issues_found,
            security_score=report.security_score,
            maintainability_score=report.maintainability_score,
            issues=[
                Issue(
                    file_path=issue.file_path,
                    line_number=issue.line_number,
                    category=issue.category,
                    severity=issue.severity,
                    description=issue.description,
                    suggestion=issue.suggestion,
                    confidence=issue.confidence,
                
                ) for issue in report.issues
            ]
        )


