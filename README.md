# Code Quality Intelligence Agent

A powerful tool to analyze code repositories, detect quality issues, generate detailed reports, and provide interactive Q&A for developers. Supports multiple programming languages, with a focus on modularity and developer-friendly insights.

## Features

- **Code Analysis**: Accepts local files, folders, or GitHub repo URLs; supports Python and JavaScript ( extensible to other languages).
- **Quality Issue Detection**: Identifies security vulnerabilities, code duplication, and complexity issues using AST parsing.
- **Detailed Reports**: Generates comprehensive reports with issue details, line numbers, categories, and fix suggestions, prioritized by severity.
- **Interactive Q&A**: Allows natural-language queries about the codebase, powered by a vector database for semantic search.
- **Web Deployment (Bonus)**: Analyze GitHub repos via a lightweight web UI.
- **RAG Integration (Super Stretch)**: Uses Retrieval-Augmented Generation for large codebases, embedding code chunks in a vector DB for efficient Q&A.
- **Visualizations (Super Stretch)**: Includes dependency graphs and issue severity charts.

## Setup

### Prerequisites

- Python 3.8+
- Node.js (for JavaScript analysis)
- Git
- Dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/username/obfuscated-project-name.git
   cd obfuscated-project-name
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install JavaScript dependencies (for web UI):

   ```bash
   npm install
   ```

### Configuration

- **Vector DB**: Uses FAISS for local vector storage. Configure in `config.yaml`.

- **GitHub API (Optional)**: Add a GitHub token in `.env` for web-based repo analysis:

  ```env
  GITHUB_TOKEN=your_token_here
  ```

## Usage

### CLI

Analyze a local codebase:

```bash
python -m code_analyzer analyze /path/to/code
```

Analyze a GitHub repository:

```bash
python -m code_analyzer analyze --github https://github.com/username/repo
```

Start interactive Q&A:

```bash
python -m code_analyzer qa /path/to/code
```

### Web UI

1. Start the web server:

   ```bash
   npm run start
   ```

2. Open `http://localhost:3000` in your browser.

3. Enter a GitHub repo URL or upload a local codebase.

### Example Output

**Report** (saved as `report.md`):

### # Code Quality Report\\

### Summary\\

- Files Analyzed: 25\\
- Issues Found: 12 (3 High, 5 Medium, 4 Low)\\

### Issues\\

1. \*\*Security Vulnerability\*\* (High) - File: \`app.py\`, Line: 42\\

Issue: Hardcoded API key detected.\\

- Suggestion: Use environment variables.\\

1. \*\*Code Duplication\*\* (Medium) - File: \`utils.js\`, Lines: 15-20\\
   - Issue: Repeated function logic.\\
   - Suggestion: Refactor into a reusable function.\
     ...

### **Interactive Q&A**:

### &gt; What does \`utils.js\` do?\\

### The \`utils.js\` file contains helper functions for data processing, including \`parseData\` and \`formatOutput\`. It’s used by \`main.js\` for preprocessing inputs.\\

&gt; Where is \`parseData\` defined?\
\`parseData\` is defined in \`utils.js\` at line 10.

### Architecture Diagram

![Architecture Diagram](images/arch.png)