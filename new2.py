#!/usr/bin/env python3
"""
Enhanced Codebase Analyzer CLI - Advanced tool to analyze codebases and identify issues
Uses LangChain with LCEL syntax, multiple analysis modes, and advanced features
"""

import os
import argparse
import json
import asyncio
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import fnmatch
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import subprocess
import re
from dotenv import load_dotenv

from rag import SemanticCodeAnalyzer


from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Custom analyzers
import ast
import tokenize
import io

load_dotenv()
class CodeIssue(BaseModel):
    """Enhanced schema for code issues"""
    file_path: str = Field(description="Path to the file with issues")
    line_number: int = Field(description="Line number where issue occurs", default=0)
    end_line_number: int = Field(description="End line number for multi-line issues", default=0)
    issue_type: str = Field(description="Type of issue",default="")
    category: str = Field(description="Category: security, performance, maintainability, style, bug, documentation")
    severity: str = Field(description="Severity: critical, high, medium, low")
    confidence: float = Field(description="Confidence score 0-1", default=0.8)
    description: str = Field(description="Description of the issue")
    suggestion: str = Field(description="Suggested fix or improvement")
    code_snippet: str = Field(description="Relevant code snippet", default="")
    rule_id: str = Field(description="Rule identifier", default="")
    tags: List[str] = Field(description="Additional tags", default_factory=list)
    impact: str = Field(description="Business/technical impact", default="")
    effort: str = Field(description="Estimated effort to fix: low, medium, high", default="medium")


class FileMetrics(BaseModel):
    """Metrics for individual files"""
    file_path: str
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    test_coverage: float = 0.0
    duplicate_lines: int = 0
    dependencies: List[str] = Field(default_factory=list)
    security_hotspots: int = 0


class CodebaseReport(BaseModel):
    """Enhanced schema for codebase analysis report"""
    timestamp: str = Field(description="Analysis timestamp")
    project_name: str = Field(description="Project name")
    summary: str = Field(description="Overall summary of the codebase analysis")
    total_files_analyzed: int = Field(description="Number of files analyzed")
    total_lines_of_code: int = Field(description="Total lines of code")
    total_issues_found: int = Field(description="Total number of issues found")
    issues_by_category: Dict[str, int] = Field(description="Count of issues by category")
    issues_by_severity: Dict[str, int] = Field(description="Count of issues by severity")
    technical_debt_ratio: float = Field(description="Technical debt ratio", default=0.0)
    maintainability_score: float = Field(description="Overall maintainability score", default=0.0)
    security_score: float = Field(description="Security score", default=0.0)
    file_metrics: List[FileMetrics] = Field(description="Metrics for each file")
    issues: List[CodeIssue] = Field(description="List of all issues found")
    trends: Dict[str, Any] = Field(description="Trends from historical data", default_factory=dict)
    recommendations: List[str] = Field(description="High-level recommendations")


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    max_files: int = 50
    max_file_size: int = 20000  # characters
    enable_security_scan: bool = True
    enable_complexity_analysis: bool = True
    enable_duplicate_detection: bool = True
    enable_dependency_analysis: bool = True
    enable_test_analysis: bool = True
    confidence_threshold: float = 0.6
    parallel_workers: int = 4
        # NEW SEMANTIC ANALYSIS OPTIONS
    enable_semantic_analysis: bool = True
    semantic_collection_name: str = "codebase_analysis"
    semantic_persist_dir: str = "./chroma_db"
    enable_code_qa: bool = True


class StaticAnalyzer:
    """Static analysis utilities"""
    
    @staticmethod
    def analyze_python_complexity(code: str) -> Dict[str, Any]:
        """Analyze Python code complexity"""
        try:
            tree = ast.parse(code)
            complexity_visitor = ComplexityVisitor()
            complexity_visitor.visit(tree)
            
            return {
                "cyclomatic_complexity": complexity_visitor.complexity,
                "functions": complexity_visitor.functions,
                "classes": complexity_visitor.classes,
                "imports": complexity_visitor.imports
            }
        except:
            return {"cyclomatic_complexity": 0, "functions": [], "classes": [], "imports": []}
    
    @staticmethod
    def detect_security_patterns(code: str, language: str) -> List[Dict[str, Any]]:
        """Detect common security patterns"""
        patterns = []
        
        # SQL Injection patterns
        sql_patterns = [
            r"execute\s*\(\s*['\"].*%.*['\"]",
            r"cursor\.execute\s*\(\s*['\"].*\+",
            r"query\s*=\s*['\"].*%.*['\"]",
            r"sql.*=.*['\"].*\+.*['\"]"
        ]
        
        # XSS patterns  
        xss_patterns = [
            r"innerHTML\s*=.*\+",
            r"document\.write\s*\(",
            r"eval\s*\(",
            r"dangerouslySetInnerHTML"
        ]
        
        # Path traversal
        path_patterns = [
            r"open\s*\(.*\+.*\)",
            r"file\s*=.*\+",
            r"\.\./",
            r"\.\.\\",
        ]
        
        all_patterns = {
            "sql_injection": sql_patterns,
            "xss": xss_patterns, 
            "path_traversal": path_patterns
        }
        
        for category, pattern_list in all_patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    patterns.append({
                        "line": line_num,
                        "category": category,
                        "pattern": pattern,
                        "match": match.group()
                    })
        
        return patterns
    
    @staticmethod
    def calculate_maintainability_index(loc: int, complexity: int, halstead_volume: float = 100) -> float:
        """Calculate maintainability index"""
        if loc == 0:
            return 100
        
        # Simplified MI calculation
        mi = 171 - 5.2 * (halstead_volume / loc) - 0.23 * complexity - 16.2 * (loc / 1000)
        return max(0, min(100, mi))


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor for complexity analysis"""
    
    def __init__(self):
        self.complexity = 1
        self.functions = []
        self.classes = []
        self.imports = []
    
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)


class DatabaseManager:
    """Manage analysis history and trends"""
    
    def __init__(self, db_path: str = "analysis_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                project_name TEXT,
                total_files INTEGER,
                total_issues INTEGER,
                security_score REAL,
                maintainability_score REAL,
                report_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, report: CodebaseReport):
        """Save analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_runs 
            (timestamp, project_name, total_files, total_issues, security_score, maintainability_score, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.timestamp,
            report.project_name,
            report.total_files_analyzed,
            report.total_issues_found,
            report.security_score,
            report.maintainability_score,
            json.dumps(report.dict())
        ))
        
        conn.commit()
        conn.close()
    
    def get_trends(self, project_name: str, limit: int = 10) -> Dict[str, Any]:
        """Get trend data for project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, total_issues, security_score, maintainability_score
            FROM analysis_runs
            WHERE project_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (project_name, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {}
        
        return {
            "historical_runs": len(results),
            "issue_trend": [r[1] for r in reversed(results)],
            "security_trend": [r[2] for r in reversed(results)],
            "maintainability_trend": [r[3] for r in reversed(results)]
        }


class EnhancedCodebaseAnalyzer:
    """Enhanced codebase analyzer with advanced features"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", config: AnalysisConfig = None):
        """Initialize the enhanced analyzer"""

        self.config = config or AnalysisConfig()


          
        # Initialize semantic analyzer if enabled
        self.semantic_analyzer = SemanticCodeAnalyzer(
                collection_name=self.config.semantic_collection_name,
                persist_directory=self.config.semantic_persist_dir
            )
        self.llm = ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0.1
        )
        
        self.embeddings = OpenAIEmbeddings(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.config = config or AnalysisConfig()
        self.db_manager = DatabaseManager()
        
        # Supported file extensions with language mapping
        self.language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.jsx': 'React JSX', '.tsx': 'React TSX', '.java': 'Java',
            '.cpp': 'C++', '.c': 'C', '.cs': 'C#', '.php': 'PHP',
            '.rb': 'Ruby', '.go': 'Go', '.rs': 'Rust', '.swift': 'Swift',
            '.kt': 'Kotlin', '.scala': 'Scala', '.sh': 'Shell Script',
            '.sql': 'SQL', '.html': 'HTML', '.css': 'CSS', '.json': 'JSON',
            '.yaml': 'YAML', '.yml': 'YAML', '.xml': 'XML'
        }
        
        # Enhanced ignore patterns
        self.ignore_patterns = [
            '*.pyc', '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            '*.min.js', '*.min.css', 'dist', 'build', '.DS_Store', '*.log',
            'coverage', '.coverage', '.nyc_output', '.pytest_cache',
            '*.egg-info', '.idea', '.vscode', 'target', 'bin', 'obj'
        ]
        
        self._setup_enhanced_chains()
    
    def _setup_enhanced_chains(self):
        """Setup enhanced analysis chains"""
        
        # Multi-aspect analysis prompt
        comprehensive_analysis_prompt = ChatPromptTemplate.from_template("""
        You are an expert senior software engineer and security analyst. Perform a comprehensive analysis of this code.
        
        File: {file_path}
        Language: {language}
        Static Analysis Results: {static_analysis}
        Code Content:
        {code_content}
        
        Analyze for:
        
        1. SECURITY VULNERABILITIES:
           - SQL injection, XSS, CSRF vulnerabilities
           - Authentication/authorization flaws
           - Input validation issues
           - Cryptographic weaknesses
           - Insecure data handling
        
        2. PERFORMANCE ISSUES:
           - Inefficient algorithms (O(n²) when O(n) possible)
           - Memory leaks and excessive allocations
           - Database query optimization opportunities
           - Caching opportunities
           - Async/await usage issues
        
        3. CODE QUALITY & MAINTAINABILITY:
           - Complex functions (>15 lines, high cyclomatic complexity)
           - Code duplication
           - Poor naming conventions
           - Missing error handling
           - Tight coupling, low cohesion
        
        4. BUGS & LOGIC ERRORS:
           - Null pointer exceptions
           - Off-by-one errors
           - Race conditions
           - Resource leaks
           - Exception handling issues
        
        5. DOCUMENTATION & TESTING:
           - Missing documentation
           - Unclear variable/function names
           - Missing unit tests
           - Poor code comments
        
        For each issue provide ALL required fields:
        - line_number: Exact line number (integer, use 1 if unknown)
        - issue_type: Type of issue (string, required)
        - category: One of: security, performance, maintainability, bug, documentation, style
        - severity: One of: critical, high, medium, low
        - confidence: Confidence score 0.0-1.0 (float)
        - description: Clear description (string, required)
        - suggestion: Specific suggestion (string, required)
        - code_snippet: Relevant code snippet (string, can be empty)
        - effort: Estimated effort: low, medium, high
        - impact: Business/technical impact (string)
        - rule_id: Rule identifier (string, can be auto-generated)
        - tags: Array of relevant tags (array, can be empty)
        
        IMPORTANT: Return ONLY a valid JSON array. Each object must have ALL fields above.
        
        Example format:
        [
          {{
            "line_number": 15,
            "issue_type": "security_vulnerability",
            "category": "security", 
            "severity": "high",
            "confidence": 0.9,
            "description": "Potential SQL injection vulnerability in database query",
            "suggestion": "Use parameterized queries instead of string concatenation",
            "code_snippet": "query = 'SELECT * FROM users WHERE id = ' + user_id",
            "effort": "medium",
            "impact": "Could allow unauthorized database access",
            "rule_id": "SQL_INJECTION_001",
            "tags": ["sql", "injection", "database"]
          }}
        ]
        
        If no issues found, return: []
        """)
        
        
        # Architecture analysis prompt
        architecture_prompt = ChatPromptTemplate.from_template("""
        Based on the codebase analysis results, provide architectural recommendations:
        
        Project Structure: {project_structure}
        File Dependencies: {dependencies}
        Analysis Summary: {analysis_summary}
        
        Provide recommendations for:
        1. Code organization and modularity
        2. Design patterns that could be applied
        3. Refactoring opportunities
        4. Testing strategy improvements
        5. Security architecture enhancements
        6. Performance optimization strategies
        7. Technical debt reduction plan
        
        Focus on actionable, high-impact recommendations.
        """)
        
        # Setup parsers
        json_parser = JsonOutputParser()
        str_parser = StrOutputParser()
        
        # Create enhanced chains
        self.comprehensive_analysis_chain = (
            comprehensive_analysis_prompt 
            | self.llm 
            | json_parser
        )
        
        self.architecture_chain = (
            architecture_prompt
            | self.llm 
            | str_parser
        )
        
        # Similarity analysis for duplicate detection
        self.similarity_chain = RunnableLambda(self._analyze_code_similarity)
    
    def _analyze_code_similarity(self, code_chunks: List[str]) -> List[Dict[str, Any]]:
        """Analyze code similarity for duplicate detection"""
        if len(code_chunks) < 2:
            return []
        
        # Create embeddings for code chunks
        embeddings = self.embeddings.embed_documents(code_chunks)
        
        duplicates = []
        threshold = 0.9  # Similarity threshold
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                if similarity > threshold:
                    duplicates.append({
                        "chunk1_index": i,
                        "chunk2_index": j,
                        "similarity": similarity,
                        "code1": code_chunks[i][:200],
                        "code2": code_chunks[j][:200]
                    })
        
        return duplicates
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _collect_files(self, directory: Path) -> List[Path]:
        """Enhanced file collection with better filtering"""
        files = []
        
        for root, dirs, filenames in os.walk(directory):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in self.ignore_patterns)]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                if (file_path.suffix in self.language_map and 
                    not self._should_ignore_file(file_path) and
                    file_path.stat().st_size < self.config.max_file_size * 10):  # Size check
                    files.append(file_path)
        
        return files[:self.config.max_files]  # Limit total files
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Enhanced file filtering"""
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(str(file_path), pattern):
                return True
        return False
    
    def _read_file_safely(self, file_path: Path) -> str:
        """Enhanced file reading with encoding detection"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to other encodings
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except Exception:
                return f"Error: Could not decode file {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
        # Limit content size
        if len(content) > self.config.max_file_size:
            content = content[:self.config.max_file_size] + "\n... [File truncated]"
        
        return content
    
    def _perform_static_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Perform static analysis on file"""
        language = self.language_map.get(file_path.suffix, 'Unknown')
        static_results = {}
        
        # Line count
        lines = content.split('\n')
        static_results['lines_of_code'] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        # Language-specific analysis
        if language == 'Python' and self.config.enable_complexity_analysis:
            complexity_data = StaticAnalyzer.analyze_python_complexity(content)
            static_results.update(complexity_data)
        
        # Security pattern detection
        if self.config.enable_security_scan:
            security_patterns = StaticAnalyzer.detect_security_patterns(content, language)
            static_results['security_patterns'] = security_patterns
        
        return static_results
    
    async def analyze_file_comprehensive(self, file_path: Path, base_path: Path) -> Dict[str, Any]:
        """Comprehensive file analysis"""
        try:
            content = self._read_file_safely(file_path)
            relative_path = file_path.relative_to(base_path)
            language = self.language_map.get(file_path.suffix, 'Unknown')
            
            print(f"🔍 Analyzing: {relative_path}")
            
            # Perform static analysis
            static_analysis = self._perform_static_analysis(file_path, content)
            
            # LLM-based analysis
            result = await self.comprehensive_analysis_chain.ainvoke({
                "file_path": str(relative_path),
                "language": language,
                "static_analysis": json.dumps(static_analysis, indent=2),
                "code_content": content
            })
            
            # Enhance results with static analysis data
            for issue in result:
                issue["file_path"] = str(relative_path)
                # Add confidence scoring based on static analysis
                if any(pattern['line'] == issue.get('line_number', 0) 
                      for pattern in static_analysis.get('security_patterns', [])):
                    issue['confidence'] = min(1.0, issue.get('confidence', 0.8) + 0.2)
            
            # Calculate file metrics
            complexity = static_analysis.get('cyclomatic_complexity', 1)
            loc = static_analysis.get('lines_of_code', 1)
            maintainability = StaticAnalyzer.calculate_maintainability_index(loc, complexity)
            
            file_metrics = FileMetrics(
                file_path=str(relative_path),
                lines_of_code=loc,
                complexity_score=float(complexity),
                maintainability_index=maintainability,
                dependencies=static_analysis.get('imports', []),
                security_hotspots=len(static_analysis.get('security_patterns', []))
            )
            
            return {
                'issues': result,
                'metrics': file_metrics,
                'static_analysis': static_analysis
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {str(e)}")
            return {
                'issues': [{
                    "file_path": str(file_path.relative_to(base_path)),
                    "line_number": 0,
                    "category": "analysis_error",
                    "severity": "low",
                    "confidence": 1.0,
                    "description": f"Analysis failed: {str(e)}",
                    "suggestion": "Check file encoding and syntax",
                    "code_snippet": "",
                    "rule_id": "ANALYSIS_ERROR",
                    "effort": "low",
                    "impact": "Analysis could not be completed"
                }],
                'metrics': FileMetrics(
                    file_path=str(file_path.relative_to(base_path)),
                    lines_of_code=0,
                    complexity_score=0.0,
                    maintainability_index=0.0
                ),
                'static_analysis': {}
            }
    
    def analyze_codebase(self, directory: str, output_file: str = None, 
                        project_name: str = None) -> CodebaseReport:
        """Enhanced codebase analysis"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        project_name = project_name or directory_path.name
        print(f"🚀 Starting comprehensive analysis for: {project_name}")
        
        # Collect files
        files = self._collect_files(directory_path)
        semantic_files = [f for f in files if f.suffix in ['.py', '.js', '.jsx', '.ts', '.tsx']]
        
        print(f"📁 Found {len(files)} files to analyze ({len(semantic_files)} for semantic analysis)")
        
        if not files:
            return self._create_empty_report(project_name)
        

        if self.config.enable_semantic_analysis and self.semantic_analyzer and semantic_files:
         print(f"🧠 Starting semantic indexing of {len(semantic_files)} files...")
        try:
            # Clear existing collection if requested
            if getattr(self.config, 'clear_semantic_db', False):
                print("🗑️ Clearing existing semantic database...")
                self.semantic_analyzer.clear_collection()
            
            # Index the codebase
            indexed_chunks = self.semantic_analyzer.index_codebase(semantic_files, directory_path)
            print(f"✅ Semantic indexing complete: {indexed_chunks} chunks indexed")
        except Exception as e:
            print(f"⚠️ Semantic indexing failed: {e}")
        
        # Enhanced duplicate detection using semantic similarity
        
        
        # Analyze files with progress tracking
        all_results = []
        
        async def analyze_batch():
            semaphore = asyncio.Semaphore(self.config.parallel_workers)
            
            async def analyze_with_semaphore(file_path):
                async with semaphore:
                    return await self.analyze_file_comprehensive(file_path, directory_path)
            
            tasks = [analyze_with_semaphore(fp) for fp in files]
            return await asyncio.gather(*tasks)
        
        try:
            all_results = asyncio.run(analyze_batch())
        except Exception as e:
            print(f"⚠️ Async analysis failed, falling back to sync: {e}")
            # Fallback to synchronous
            for file_path in files:
                result = asyncio.run(self.analyze_file_comprehensive(file_path, directory_path))
                all_results.append(result)
        
        # Process results
        all_issues = []
        all_metrics = []
        
        for result in all_results:
            all_issues.extend(result['issues'])
            all_metrics.append(result['metrics'])
        
        print(f"📊 Processed {len(all_metrics)} file metrics")
        if all_metrics:
            sample_metric = all_metrics[0]
            print(f"📋 Sample metric: {sample_metric.file_path} - LOC: {sample_metric.lines_of_code}, Complexity: {sample_metric.complexity_score:.1f}")
        
        # Filter by confidence threshold
        filtered_issues = [
            issue for issue in all_issues 
            if issue.get('confidence', 0.8) >= self.config.confidence_threshold
        ]
        
        print(f"✅ Analysis complete! Found {len(filtered_issues)} high-confidence issues")
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(
            project_name, files, filtered_issues, all_metrics, directory_path
        )
        
        # Save to database
        self.db_manager.save_analysis(report)
        
        # Save report file
        if output_file:
            self._save_report(report, output_file)
        
        return report
    

    def _find_semantic_duplicates(self) -> List[Dict[str, Any]]:
        """Find semantic duplicates using ChromaDB similarity"""
        try:
            # Get all chunks from the collection
            sample_results = self.semantic_analyzer.collection.query(
                query_texts=["code function class"],
                n_results=min(1000, self.semantic_analyzer.collection.count())
            )
            
            duplicates = []
            threshold = 0.95  # High similarity threshold for duplicates
            
            if not sample_results['documents'] or not sample_results['documents'][0]:
                return duplicates
            
            docs = sample_results['documents'][0]
            metadatas = sample_results['metadatas'][0]
            
            # Compare each chunk with others
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    if metadatas[i]['file_path'] != metadatas[j]['file_path']:  # Different files only
                        # Use semantic search to find similarity
                        similar_results = self.semantic_analyzer.collection.query(
                            query_texts=[docs[i]],
                            n_results=1,
                            where={"file_path": metadatas[j]['file_path']}
                        )
                        
                        if similar_results['distances'] and similar_results['distances'][0]:
                            distance = similar_results['distances'][0][0]
                            similarity = 1 - distance
                            
                            if similarity > threshold:
                                duplicates.append({
                                    "file1": metadatas[i]['file_path'],
                                    "file2": metadatas[j]['file_path'],
                                    "similarity": similarity,
                                    "chunk_type": metadatas[i]['chunk_type'],
                                    "lines1": f"{metadatas[i]['start_line']}-{metadatas[i]['end_line']}",
                                    "lines2": f"{metadatas[j]['start_line']}-{metadatas[j]['end_line']}"
                                })
            
            return duplicates[:10]  # Limit to top 10 duplicates
            
        except Exception as e:
            print(f"Error finding semantic duplicates: {e}")
            return []
    
    def ask_codebase_question(self, question: str) -> Dict[str, Any]:
        """Ask questions about the codebase using RAG"""
        if not self.semantic_analyzer:
            return {"error": "Semantic analysis not enabled"}
        
        try:
            result = self.semantic_analyzer.answer_code_question(question)
            
            # Enhance with LLM-powered answer generation
            if result["relevant_chunks"]:
                context = result["context"]
                
                qa_prompt = f"""
                Based on the following code context from the codebase, answer the user's question:
                
                Question: {question}
                
                Code Context:
                {context}
                
                Provide a comprehensive answer that:
                1. Directly addresses the question
                2. References specific code examples when relevant
                3. Explains the reasoning behind the code patterns
                4. Suggests improvements if applicable
                
                Answer:
                """
                
                try:
                    llm_response = self.llm.invoke(qa_prompt)
                    result["answer"] = llm_response.content
                except Exception as e:
                    result["answer"] = f"Found relevant code but couldn't generate answer: {e}"
            
            return result
            
        except Exception as e:
            return {"error": f"Error processing question: {e}"}
    
    def search_codebase(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search codebase semantically"""
        if not self.semantic_analyzer:
            return []
        
        return self.semantic_analyzer.semantic_search(
            query=query,
            n_results=20,
            filter_params=filters or {}
        )
    
    def get_similar_functions(self, file_path: str, function_name: str) -> List[Dict[str, Any]]:
        """Find functions similar to a specific function"""
        if not self.semantic_analyzer:
            return []
        
        # Get the target function
        context_results = self.semantic_analyzer.get_code_context(file_path, function_name)
        
        if not context_results:
            return []
        
        target_function = context_results[0]["content"]
        
        # Find similar functions
        similar = self.semantic_analyzer.find_similar_code(target_function, n_results=10)
        
        # Filter out the original function
        return [s for s in similar if s["metadata"]["file_path"] != file_path or 
                s["metadata"]["function_name"] != function_name]
    
    def get_semantic_stats(self) -> Dict[str, Any]:
        """Get semantic analysis statistics"""
        if not self.semantic_analyzer:
            return {"error": "Semantic analysis not enabled"}
        
        return self.semantic_analyzer.get_collection_stats()



    def _create_empty_report(self, project_name: str) -> CodebaseReport:
        """Create empty report when no files found"""
        return CodebaseReport(
            timestamp=datetime.now().isoformat(),
            project_name=project_name,
            summary="No supported files found for analysis",
            total_files_analyzed=0,
            total_lines_of_code=0,
            total_issues_found=0,
            issues_by_category={},
            issues_by_severity={},
            technical_debt_ratio=0.0,
            maintainability_score=100.0,
            security_score=100.0,
            file_metrics=[],
            issues=[],
            recommendations=["Add supported source code files to enable analysis"]
        )
    
    def _generate_comprehensive_report(self, project_name: str, files: List[Path], 
                                     issues: List[Dict], metrics: List[FileMetrics],
                                     directory_path: Path) -> CodebaseReport:
        """Generate comprehensive analysis report"""
        
        # Calculate aggregated metrics
        total_loc = sum(m.lines_of_code for m in metrics)
        avg_complexity = sum(m.complexity_score for m in metrics) / len(metrics) if metrics else 0
        avg_maintainability = sum(m.maintainability_index for m in metrics) / len(metrics) if metrics else 100
        
        # Count issues by category and severity
        category_counts = {}
        severity_counts = {}
        
        for issue in issues:
            category = issue.get('category', 'unknown')
            severity = issue.get('severity', 'unknown')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate scores
        security_issues = category_counts.get('security', 0)
        critical_issues = severity_counts.get('critical', 0)
        high_issues = severity_counts.get('high', 0)
        
        security_score = max(0, 100 - (security_issues * 20) - (critical_issues * 10))
        technical_debt_ratio = min(100, (len(issues) / total_loc * 1000) if total_loc > 0 else 0)
        
        # Get trends
        trends = self.db_manager.get_trends(project_name)
        
        # Generate summary and recommendations
        summary_data = {
            "total_files": len(files),
            "total_loc": total_loc,
            "total_issues": len(issues),
            "avg_complexity": avg_complexity,
            "security_score": security_score,
            "maintainability_score": avg_maintainability,
            "critical_issues": critical_issues,
            "high_issues": high_issues
        }
        
        try:
            # Generate AI summary
            summary_prompt = f"""
            Codebase Analysis Summary for {project_name}:
            
            Files: {len(files)}
            Lines of Code: {total_loc:,}
            Issues Found: {len(issues)}
            Average Complexity: {avg_complexity:.1f}
            Security Score: {security_score:.1f}/100
            Maintainability: {avg_maintainability:.1f}/100
            
            Critical Issues: {critical_issues}
            High Priority Issues: {high_issues}
            
            Top Issue Categories: {dict(list(category_counts.items())[:3])}
            
            Provide a concise executive summary and key insights.
            """
            
            summary = self.llm.invoke(summary_prompt).content
            
        except Exception as e:
            summary = f"Analysis completed for {project_name} with {len(issues)} issues found across {len(files)} files."
        
        # Generate recommendations
        recommendations = self._generate_recommendations(category_counts, severity_counts, avg_complexity, security_score)
        
        return CodebaseReport(
            timestamp=datetime.now().isoformat(),
            project_name=project_name,
            summary=summary,
            total_files_analyzed=len(files),
            total_lines_of_code=total_loc,
            total_issues_found=len(issues),
            issues_by_category=category_counts,
            issues_by_severity=severity_counts,
            technical_debt_ratio=technical_debt_ratio,
            maintainability_score=avg_maintainability,
            security_score=security_score,
            file_metrics=metrics,
            issues=[CodeIssue(**issue) for issue in issues],
            trends=trends,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, categories: Dict, severities: Dict, 
                                complexity: float, security_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if severities.get('critical', 0) > 0:
            recommendations.append(f"🚨 URGENT: Address {severities['critical']} critical issues immediately")
        
        if security_score < 70:
            recommendations.append("🔒 Security Review: Implement security code review process")
        
        if categories.get('performance', 0) > 5:
            recommendations.append("⚡ Performance Audit: Review and optimize performance bottlenecks")
        
        if complexity > 10:
            recommendations.append("🔄 Refactoring: Break down complex functions and classes")
        
        if categories.get('maintainability', 0) > 10:
            recommendations.append("🛠️ Technical Debt: Allocate time for refactoring and cleanup")
        
        if categories.get('documentation', 0) > 5:
            recommendations.append("📚 Documentation: Improve code documentation and comments")
        
        recommendations.append("✅ Implement automated code quality checks in CI/CD pipeline")
        
        return recommendations
    
    def _save_report(self, report: CodebaseReport, output_file: str):
        """Save report in multiple formats"""
        base_path = Path(output_file).stem
        
        # JSON report
        with open(f"{base_path}.json", 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        # HTML report
        self._generate_html_report(report, f"{base_path}.html")
        
        # CSV for issues
        self._generate_csv_report(report, f"{base_path}_issues.csv")
        
        print(f"📊 Reports saved: {base_path}.json, {base_path}.html, {base_path}_issues.csv")
    
    def _generate_html_report(self, report: CodebaseReport, output_file: str):
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Codebase Analysis Report - {project_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .issue {{ border-left: 4px solid #dc3545; padding: 15px; margin: 10px 0; background: #f8f9fa; }}
                .issue.high {{ border-color: #ff6b6b; }}
                .issue.medium {{ border-color: #ffd93d; }}
                .issue.low {{ border-color: #6bcf7f; }}
                .critical {{ border-color: #dc3545; background: #fff5f5; }}
                .recommendations {{ background: #e8f5e8; padding: 20px; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Codebase Analysis Report</h1>
                <h2>{project_name}</h2>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{total_files}</div>
                    <div>Files Analyzed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_loc:,}</div>
                    <div>Lines of Code</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_issues}</div>
                    <div>Issues Found</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{security_score:.1f}</div>
                    <div>Security Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{maintainability_score:.1f}</div>
                    <div>Maintainability</div>
                </div>
            </div>
            
            <h3>📊 Summary</h3>
            <p>{summary}</p>
            
            <h3>🎯 Recommendations</h3>
            <div class="recommendations">
                {recommendations_html}
            </div>
            
            <h3>⚠️ Critical & High Priority Issues</h3>
            {critical_issues_html}
            
            <h3>📈 Issues by Category</h3>
            <table>
                <tr><th>Category</th><th>Count</th></tr>
                {category_table}
            </table>
            
            <h3>📋 All Issues</h3>
            {all_issues_html}
        </body>
        </html>
        """
        
        # Prepare data
        critical_issues = [issue for issue in report.issues if issue.severity == 'critical']
        high_issues = [issue for issue in report.issues if issue.severity == 'high']
        
        recommendations_html = "<ul>" + "".join([f"<li>{rec}</li>" for rec in report.recommendations]) + "</ul>"
        
        critical_issues_html = ""
        for issue in critical_issues + high_issues:
            critical_issues_html += f"""
            <div class="issue {issue.severity}">
                <h4>{issue.file_path}:{issue.line_number}</h4>
                <p><strong>{issue.issue_type}</strong> - {issue.severity.upper()}</p>
                <p>{issue.description}</p>
                <p><strong>Suggestion:</strong> {issue.suggestion}</p>
            </div>
            """
        
        category_table = ""
        for category, count in report.issues_by_category.items():
            category_table += f"<tr><td>{category.capitalize()}</td><td>{count}</td></tr>"
        
        all_issues_html = ""
        for issue in report.issues[:20]:  # First 20 issues
            all_issues_html += f"""
            <div class="issue {issue.severity}">
                <strong>{issue.file_path}:{issue.line_number}</strong> - 
                {issue.issue_type} ({issue.severity}) - {issue.description}
            </div>
            """
        
        html_content = html_template.format(
            project_name=report.project_name,
            timestamp=report.timestamp,
            total_files=report.total_files_analyzed,
            total_loc=report.total_lines_of_code,
            total_issues=report.total_issues_found,
            security_score=report.security_score,
            maintainability_score=report.maintainability_score,
            summary=report.summary,
            recommendations_html=recommendations_html,
            critical_issues_html=critical_issues_html,
            category_table=category_table,
            all_issues_html=all_issues_html
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_csv_report(self, report: CodebaseReport, output_file: str):
        """Generate CSV report for issues"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_path', 'line_number', 'category', 'severity', 'confidence', 
                         'description', 'suggestion', 'effort', 'impact', 'rule_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for issue in report.issues:
                writer.writerow({
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'category': issue.category,
                    'severity': issue.severity,
                    'confidence': issue.confidence,
                    'description': issue.description,
                    'suggestion': issue.suggestion,
                    'effort': issue.effort,
                    'impact': issue.impact,
                    'rule_id': issue.rule_id
                })


def print_enhanced_report(report: CodebaseReport):
    """Print enhanced formatted report to console"""
    print("\n" + "="*100)
    print(f"🔍 CODEBASE ANALYSIS REPORT - {report.project_name}")
    print("="*100)
    
    # Key metrics
    print(f"""
📊 KEY METRICS:
   Files Analyzed: {report.total_files_analyzed:,}
   Lines of Code: {report.total_lines_of_code:,}
   Issues Found: {report.total_issues_found:,}
   Security Score: {report.security_score:.1f}/100
   Maintainability: {report.maintainability_score:.1f}/100
   Technical Debt: {report.technical_debt_ratio:.1f}%
""")
    
    # Issues by severity
    if report.issues_by_severity:
        print("⚠️  ISSUES BY SEVERITY:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = report.issues_by_severity.get(severity, 0)
            if count > 0:
                emoji = {'critical': '🚨', 'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                print(f"   {emoji} {severity.capitalize()}: {count}")
    
    # Issues by category
    if report.issues_by_category:
        print("\n📂 ISSUES BY CATEGORY:")
        for category, count in sorted(report.issues_by_category.items(), key=lambda x: x[1], reverse=True):
            emoji = {'security': '🔒', 'performance': '⚡', 'maintainability': '🛠️', 
                    'bug': '🐛', 'documentation': '📚', 'style': '✨'}.get(category, '📋')
            print(f"   {emoji} {category.capitalize()}: {count}")
    
    # Trends
    if report.trends and report.trends.get('historical_runs', 0) > 1:
        print(f"\n📈 TRENDS (Last {report.trends['historical_runs']} analyses):")
        issue_trend = report.trends.get('issue_trend', [])
        if len(issue_trend) >= 2:
            change = issue_trend[-1] - issue_trend[-2]
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   {direction} Issues: {change:+d} since last analysis")
    
    print(f"\n💡 SUMMARY:\n{report.summary}")
    
    # Recommendations
    if report.recommendations:
        print("\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Critical issues
    critical_issues = [issue for issue in report.issues if issue.severity == 'critical']
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
        for issue in critical_issues[:5]:  # Show first 5 critical
            print(f"   📍 {issue.file_path}:{issue.line_number}")
            print(f"      {issue.description}")
            print(f"      💡 {issue.suggestion}")
            print()
    
    # High priority issues
    high_issues = [issue for issue in report.issues if issue.severity == 'high']
    if high_issues and len(critical_issues) < 5:
        remaining = 5 - len(critical_issues)
        print(f"\n🔴 HIGH PRIORITY ISSUES:")
        for issue in high_issues[:remaining]:
            print(f"   📍 {issue.file_path}:{issue.line_number}")
            print(f"      {issue.description}")
            print()
    
    # File metrics summary
    if report.file_metrics:
        print(f"\n📁 FILE METRICS:")
        high_complexity = [fm for fm in report.file_metrics if fm.complexity_score > 10]
        low_maintainability = [fm for fm in report.file_metrics if fm.maintainability_index < 70]
        
        if high_complexity:
            print(f"   🔄 High Complexity Files: {len(high_complexity)}")
            for fm in high_complexity[:3]:
                print(f"      • {fm.file_path} (complexity: {fm.complexity_score:.1f})")
        
        if low_maintainability:
            print(f"   🛠️ Low Maintainability: {len(low_maintainability)}")
    
    print("\n" + "="*100)
    print(f"✅ Analysis completed at {report.timestamp}")
    print("="*100)

def create_config_from_args(args) -> AnalysisConfig:
    """Create analysis config from command line arguments"""
    return AnalysisConfig(
        max_files=args.max_files,
        max_file_size=args.max_file_size,
        enable_security_scan=args.enable_security,
        enable_complexity_analysis=args.enable_complexity,
        enable_duplicate_detection=args.enable_duplicates,
        enable_dependency_analysis=args.enable_dependencies,
        enable_test_analysis=args.enable_tests,
        confidence_threshold=args.confidence_threshold,
        parallel_workers=args.parallel_workers,
        enable_semantic_analysis=getattr(args, 'enable_semantic', True) or Any ,
        semantic_collection_name=getattr(args, 'semantic_collection', 'codebase_analysis') or Any ,
        semantic_persist_dir=getattr(args, 'semantic_db_path', './chroma_db') or Any,
        enable_code_qa=True
    )



def main():
    parser = argparse.ArgumentParser(description="Enhanced Codebase Analysis Tool")
    parser.add_argument("directory", help="Directory path to analyze")
    parser.add_argument("-o", "--output", help="Output file base name (generates .json, .html, .csv)")
    parser.add_argument("-k", "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("-p", "--project-name", help="Project name for analysis")

    #new
    parser.add_argument("--enable-semantic", action="store_true", default=True, 
                       help="Enable semantic analysis with ChromaDB")
    parser.add_argument("--semantic-collection", default="codebase_analysis",
                       help="ChromaDB collection name")
    parser.add_argument("--semantic-db-path", default="./chroma_db",
                       help="ChromaDB persistence directory")
    parser.add_argument("--clear-semantic-db", action="store_true",
                       help="Clear existing semantic database before analysis")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive Q&A mode after analysis")
    parser.add_argument("--search-query", help="Search codebase for specific query")
    
    args = parser.parse_args()
    
    # Analysis configuration
    parser.add_argument("--max-files", type=int, default=50, help="Maximum files to analyze")
    parser.add_argument("--max-file-size", type=int, default=20000, help="Maximum file size in characters")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Minimum confidence threshold")
    parser.add_argument("--parallel-workers", type=int, default=4, help="Number of parallel workers")
    
    # Feature toggles
    parser.add_argument("--enable-security", action="store_true", default=True, help="Enable security analysis")
    parser.add_argument("--enable-complexity", action="store_true", default=True, help="Enable complexity analysis")
    parser.add_argument("--enable-duplicates", action="store_true", default=True, help="Enable duplicate detection")
    parser.add_argument("--enable-dependencies", action="store_true", default=True, help="Enable dependency analysis")
    parser.add_argument("--enable-tests", action="store_true", default=True, help="Enable test analysis")
    
    # Output options
    parser.add_argument("--format", choices=['console', 'json', 'html', 'all'], default='all', 
                       help="Output format")
    parser.add_argument("--show-trends", action="store_true", help="Show historical trends")
    parser.add_argument("--export-metrics", help="Export file metrics to CSV")
    
    args = parser.parse_args()
    
    if not args.api_key and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OpenAI API key required. Use -k flag or set OPENAI_API_KEY environment variable")
        return 1
    
    try:
        config = create_config_from_args(args)
        analyzer = EnhancedCodebaseAnalyzer(api_key=args.api_key, model=args.model, config=config)
        
        # Run analysis
        report = analyzer.analyze_codebase(
            directory=args.directory, 
            output_file=args.output,
            project_name=args.project_name
        )
        
        # Display results based on format
        if args.format in ['console', 'all']:
            print_enhanced_report(report)

                # Show semantic stats
        if analyzer.semantic_analyzer:
            semantic_stats = analyzer.get_semantic_stats()
            print(f"\nSemantic Analysis Stats:")
            print(f"  Total indexed chunks: {semantic_stats.get('total_chunks', 0)}")
            if 'languages' in semantic_stats:
                print(f"  Languages: {semantic_stats['languages']}")
            if 'chunk_types' in semantic_stats:
                print(f"  Chunk types: {semantic_stats['chunk_types']}")
        
        # Handle search query
        if args.search_query:
            print(f"\nSearching for: '{args.search_query}'")
            results = analyzer.search_codebase(args.search_query)
            for i, result in enumerate(results[:5], 1):
                print(f"\n{i}. {result['metadata']['file_path']}:{result['metadata']['start_line']}")
                print(f"   Type: {result['metadata']['chunk_type']}")
                if result['metadata'].get('function_name'):
                    print(f"   Function: {result['metadata']['function_name']}")
                print(f"   {result['content'][:200]}...")
        
        # Interactive mode
        if args.interactive:
            start_interactive_mode(analyzer)

        
        
        # Export additional metrics
        if args.export_metrics:
            import csv
            with open(args.export_metrics, 'w', newline='') as csvfile:
                fieldnames = ['file_path', 'lines_of_code', 'complexity_score', 'maintainability_index', 
                             'security_hotspots', 'dependencies']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for metrics in report.file_metrics:
                    writer.writerow({
                        'file_path': metrics.file_path,
                        'lines_of_code': metrics.lines_of_code,
                        'complexity_score': metrics.complexity_score,
                        'maintainability_index': metrics.maintainability_index,
                        'security_hotspots': metrics.security_hotspots,
                        'dependencies': len(metrics.dependencies)
                    })
            print(f"📊 Metrics exported to: {args.export_metrics}")
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"   📈 {report.total_issues_found} issues found in {report.total_files_analyzed} files")
        print(f"   🔒 Security Score: {report.security_score:.1f}/100")
        print(f"   🛠️ Maintainability: {report.maintainability_score:.1f}/100")
        
        if report.total_issues_found > 0:
            critical_count = report.issues_by_severity.get('critical', 0)
            high_count = report.issues_by_severity.get('high', 0)
            if critical_count > 0:
                print(f"   🚨 {critical_count} critical issues need immediate attention!")
            elif high_count > 0:
                print(f"   🔴 {high_count} high-priority issues should be addressed soon")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
def start_interactive_mode(analyzer):
    """Start interactive Q&A mode"""
    print("\n" + "="*60)
    print("Interactive Codebase Q&A Mode")
    print("Type 'exit' to quit, 'help' for commands")
    print("="*60)
    
    while True:
        try:
            query = input("\nQ: ").strip()
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'help':
                print("""
Available commands:
- ask <question>     : Ask a question about the codebase
- search <query>     : Search for code snippets
- similar <file> <func> : Find similar functions
- stats              : Show semantic analysis statistics
- exit               : Quit interactive mode
                """)
                continue
            elif query.lower() == 'stats':
                stats = analyzer.get_semantic_stats()
                print(f"Stats: {stats}")
                continue
            elif query.startswith('search '):
                search_query = query[7:]
                results = analyzer.search_codebase(search_query)[:3]
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['metadata']['file_path']}")
                    print(f"   {result['content'][:300]}...")
                continue
            elif query.startswith('similar '):
                parts = query.split()
                if len(parts) >= 3:
                    file_path, func_name = parts[1], parts[2]
                    similar = analyzer.get_similar_functions(file_path, func_name)[:3]
                    for result in similar:
                        print(f"Similar: {result['metadata']['file_path']} - {result['metadata'].get('function_name', 'N/A')}")
                continue
            
            # Default: ask question
            if not query.startswith('ask '):
                query = 'ask ' + query
            
            question = query[4:] if query.startswith('ask ') else query
            response = analyzer.ask_codebase_question(question)
            
            if 'error' in response:
                print(f"Error: {response['error']}")
            else:
                print(f"\nAnswer: {response.get('answer', 'No answer generated')}")
                if response.get('sources'):
                    print(f"Sources: {', '.join(response['sources'])}")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")



if __name__ == "__main__":
    exit(main())