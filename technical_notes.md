# Technical Notes

## External Tools, APIs, and Libraries
- **LangChain**: Used for RAG pipeline, including document chunking and embedding generation. Simplifies integration with LLMs for Q&A.
- **Chroma Db**: Lightweight vector database for storing and querying code embeddings, enabling fast semantic searches.
- **Tree-sitter**: AST parsing library for Python and JavaScript. Provides precise structural analysis for detecting issues like code duplication and complexity.
- **GitHub API**: Used for cloning repositories in web mode. Requires a personal access token for authentication.
- **React + Tailwind CSS**: Powers the web UI for GitHub repo analysis. Hosted via CDN (e.g., `vercel` for React).
- **Fast Api**: a backend api to for the web ui using fast api and deployed through docker in render.
- **react-charts**: Generates visualizations (dependency graphs, issue severity charts) in reports and web UI.
- **python-git**: Handles local Git operations for cloning and analyzing repos.
- **dotenv**: Manages environment variables (e.g., GitHub tokens).

## Creative Features and Integrations
1. **Automated Severity Scoring**:
   - Implemented a rule-based scoring system for issue prioritization (e.g., security vulnerabilities = high, code duplication = medium).
   - Uses heuristics like cyclomatic complexity (from AST) and impact scope (e.g., global vs. local variables).
   - **Why Valuable**: Helps developers focus on critical issues first, improving productivity.
2. **Dependency Visualization**:
   - Generates interactive dependency graphs using Plotly, showing module relationships in Python (`import` statements) and JavaScript (`require`/`import`).
   - **Why Valuable**: Helps developers identify tightly coupled code or potential refactoring opportunities.
4. **Trend Tracking**:
   - Stores analysis history in a local SQLite DB to track issue trends over time (e.g., increasing complexity).
   - **Why Valuable**: Provides insights into codebase health evolution, useful for long-term maintenance.

## Implementation Details
- **AST Parsing**: Uses Tree-sitter to parse Python and JavaScript files, detecting issues like:
  - **Security**: Hardcoded secrets (e.g., API keys), insecure functions (e.g., `eval` in JS).
  - **Duplication**: Identical code blocks across files (via AST node comparison).
  - **Complexity**: High cyclomatic complexity or nested loops.
- **RAG Pipeline**: Files are chunked (500-1000 tokens), embedded using a pre-trained model (via LangChain), and stored in Chromadb . Queries are matched using cosine similarity.
- **Agentic Patterns**: Uses LangGraph for orchestrating analysis workflows (e.g., static analysis → report generation → Q&A).
- **Web UI**: Built with React and Tailwind CSS, hosted locally or deployable to Vercel. Accepts GitHub URLs and displays reports/visualizations.