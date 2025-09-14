# Architecture Notes

![Architecture Diagram](images/arch.png)

## High-Level Design Decisions

### Modular Pipeline
- **Decision**: Split the analysis into two parallel paths: static analysis (AST parsing) and semantic embedding (RAG).
  - **Rationale**: Static analysis provides precise, line-level insights for issues like security vulnerabilities and code duplication. Semantic embedding enables contextual understanding for Q&A and cross-file relationships.
  - **Trade-Offs**:
    - **Pros**: Combines precision (AST) with flexibility (RAG), supporting both detailed reports and conversational queries.
    - **Cons**: Increased complexity in managing two pipelines; requires careful synchronization for report generation.

### Language Support
- **Decision**: Support Python and JavaScript initially, with a plugin-based architecture for extensibility.
  - **Rationale**: Python and JavaScript are widely used, covering diverse use cases (backend, frontend). Plugin system allows adding languages like Java or C++ without major refactoring.
  - **Trade-Offs**:
    - **Pros**: Broad applicability and future-proofing.
    - **Cons**: Initial development focused on two languages; adding new ones requires parser implementation.

### RAG for Large Codebases
- **Decision**: Use Retrieval-Augmented Generation (RAG) with FAISS vector DB for semantic analysis.
  - **Rationale**: Chunking files and embedding them in a vector DB enables efficient similarity searches for Q&A, handling large codebases without loading everything into memory.
  - **Trade-Offs**:
    - **Pros**: Scalable for large repos; supports natural-language queries with context.
    - **Cons**: Embedding process is computationally expensive; requires tuning for optimal chunk size and embedding model.

### Web Deployment
- **Decision**: Implement a lightweight React-based web UI for GitHub repo analysis.
  - **Rationale**: Enhances accessibility for non-technical users and integrates with GitHub APIs for seamless repo cloning.
  - **Trade-Offs**:
    - **Pros**: User-friendly interface; aligns with bonus requirement.
    - **Cons**: Adds frontend development overhead; requires secure handling of GitHub tokens.

## Major Design Trade-Offs
1. **Precision vs. Scalability**:
   - AST parsing is precise but slow for large codebases. RAG is faster for Q&A but less granular. Combining both balances the trade-off but increases system complexity.
2. **Local vs. Web Deployment**:
   - Local CLI is lightweight and secure but less accessible. Web UI is user-friendly but introduces hosting and security considerations (e.g., GitHub token management).
3. **Issue Prioritization**:
   - Automated severity scoring (based on issue type and impact) improves report usability but risks false positives/negatives without manual tuning.