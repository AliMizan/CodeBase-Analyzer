
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from .schemas import AnalyzeRequest, GitCloneRequest, QARequest, ReportResponse, SearchResponse,Issue
from .services import AnalyzerService
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
analyzer_service = AnalyzerService()

@app.post("/analyze", response_model=ReportResponse)
def analyze(request: AnalyzeRequest):
    project_name = request.project_name or "default_project"
    report = analyzer_service.analyze(request.directory, project_name, request.config.dict() if request.config else {})
    return report

@app.get("/report/{project_name}", response_model=ReportResponse)
def get_report(project_name: str):
    report = analyzer_service.get_report(project_name)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@app.get("/issues/{project_name}", response_model=List[Issue])
def get_issues(project_name: str, category: str = None, severity: str = None):
    report = analyzer_service.get_report(project_name)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    filtered = report.issues
    if category:
        filtered = [i for i in filtered if i.category == category]
    if severity:
        filtered = [i for i in filtered if i.severity == severity]
    return filtered

@app.get("/summary/{project_name}", response_model=ReportResponse)
def summary(project_name: str):
    report = analyzer_service.get_report(project_name)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(content={"summary": report.summary})

@app.post("/qa/{project_name}")
def qa(project_name: str, request: QARequest):
    result = analyzer_service.ask_question(project_name, request.question)
    return result

@app.post("/analyze/github", response_model=ReportResponse)
def analyze_github(request: GitCloneRequest):
    project_name = request.project_name or Path(request.github_url).stem
    try:
        report = analyzer_service.clone_and_analyze(
            github_url=request.github_url,
            project_name=project_name,
            config_data=request.config.dict() if request.config else {}
        )
        return report
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=400, detail="Failed to clone repository")


@app.get("/search/{project_name}", response_model=List[SearchResponse])
def search(project_name: str, query: str):
    return analyzer_service.search(project_name, query)

from fastapi import FastAPI

@app.get("/projects", response_model=List[str])
def list_projects():
    return analyzer_service.list_projects()


@app.get("/trends/{project_name}")
def trends(project_name: str):
    return analyzer_service.get_trends(project_name)

@app.get("/export/{project_name}")
def export(project_name: str, format: str = "json"):
    # Implement actual file export logic
    report = analyzer_service.get_report(project_name)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    filename = f"{project_name}.{format}"
    # Here, you should write the report to file
    # For now, dummy file
    with open(filename, "w") as f:
        f.write("This is a dummy export file.")
    return FileResponse(filename, media_type="application/octet-stream", filename=filename)
