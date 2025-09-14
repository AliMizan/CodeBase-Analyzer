#!/usr/bin/env python3
"""
Semantic Code Analyzer with ChromaDB integration
Provides semantic searching and RAG Q&A capabilities for code analysis
"""

import os
import re
import ast
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken


@dataclass
class CodeChunk:
    """Represents a code chunk with metadata"""
    content: str
    file_path: str
    chunk_type: str  # function, class, import, global, comment_block
    start_line: int
    end_line: int
    language: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    dependencies: List[str] = None
    complexity_score: float = 0.0
    embedding_id: str = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.embedding_id is None:
            self.embedding_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for the chunk"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.file_path}_{self.start_line}_{self.end_line}_{content_hash}"

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format"""
        return {
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "function_name": self.function_name or "",
            "class_name": self.class_name or "",
            "dependencies": ",".join(self.dependencies),
            "complexity_score": self.complexity_score,
            "lines_count": self.end_line - self.start_line + 1
        }


class CodeChunker:
    """Smart code chunking strategies for different languages"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_code(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Main chunking method that routes to language-specific chunkers"""
        if language.lower() == 'python':
            return self._chunk_python(content, file_path)
        elif language.lower() in ['javascript', 'typescript']:
            return self._chunk_javascript(content, file_path, language)
        elif language.lower() in ['jsx', 'tsx']:
            return self._chunk_react(content, file_path, language)
        else:
            return self._chunk_generic(content, file_path, language)
    
    def _chunk_python(self, content: str, file_path: str) -> List[CodeChunk]:
        """Python-specific chunking using AST"""
        chunks = []
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
            chunks.extend(self._extract_python_chunks(tree, lines, file_path))
        except SyntaxError:
            # Fallback to generic chunking if parsing fails
            return self._chunk_generic(content, file_path, 'Python')
        
        # Add import blocks and global variables
        chunks.extend(self._extract_python_imports_and_globals(content, file_path))
        
        return self._merge_small_chunks(chunks)
    
    def _extract_python_chunks(self, tree: ast.AST, lines: List[str], file_path: str) -> List[CodeChunk]:
        """Extract Python functions and classes"""
        chunks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = self._create_python_function_chunk(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._create_python_class_chunk(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _create_python_function_chunk(self, node: ast.FunctionDef, lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Create chunk for Python function"""
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if node.end_lineno else start_line + 10
        
        # Extract function content with some context
        context_start = max(0, start_line - 2)
        context_end = min(len(lines), end_line + 2)
        
        content = '\n'.join(lines[context_start:context_end])
        
        # Calculate complexity (simplified)
        complexity = self._calculate_python_complexity(node)
        
        # Extract dependencies (function calls, imports within function)
        dependencies = self._extract_python_dependencies(node)
        
        return CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type="function",
            start_line=context_start + 1,
            end_line=context_end,
            language="Python",
            function_name=node.name,
            dependencies=dependencies,
            complexity_score=complexity
        )
    
    def _create_python_class_chunk(self, node: ast.ClassDef, lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Create chunk for Python class"""
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if node.end_lineno else start_line + 20
        
        content = '\n'.join(lines[start_line:end_line])
        
        # If class is too large, split into methods
        if self._count_tokens(content) > self.max_chunk_size:
            return self._split_large_class(node, lines, file_path)
        
        dependencies = self._extract_python_dependencies(node)
        
        return CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type="class",
            start_line=start_line + 1,
            end_line=end_line,
            language="Python",
            class_name=node.name,
            dependencies=dependencies,
            complexity_score=len([n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)])
        )
    
    def _chunk_javascript(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """JavaScript/TypeScript chunking using regex patterns"""
        chunks = []
        lines = content.split('\n')
        
        # Extract functions
        function_pattern = r'^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|class\s+\w+)'
        
        current_chunk = []
        current_start = 0
        brace_count = 0
        in_function = False
        
        for i, line in enumerate(lines):
            if re.match(function_pattern, line.strip()) and not in_function:
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append(self._create_js_chunk(current_chunk, current_start, i-1, file_path, language))
                
                current_chunk = [line]
                current_start = i
                in_function = True
                brace_count = line.count('{') - line.count('}')
            else:
                current_chunk.append(line)
                if in_function:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0 and '{' in ''.join(current_chunk):
                        # Function ended
                        chunks.append(self._create_js_chunk(current_chunk, current_start, i, file_path, language))
                        current_chunk = []
                        in_function = False
                        brace_count = 0
        
        # Add remaining content
        if current_chunk:
            chunks.append(self._create_js_chunk(current_chunk, current_start, len(lines)-1, file_path, language))
        
        return self._merge_small_chunks(chunks)
    
    def _chunk_react(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """React JSX/TSX chunking"""
        chunks = []
        lines = content.split('\n')
        
        # Extract React components
        component_pattern = r'^(?:export\s+)?(?:const\s+|function\s+)([A-Z]\w+)'
        
        current_chunk = []
        current_start = 0
        in_component = False
        brace_count = 0
        paren_count = 0
        
        for i, line in enumerate(lines):
            match = re.match(component_pattern, line.strip())
            if match and not in_component:
                # Save previous chunk
                if current_chunk:
                    chunks.append(self._create_react_chunk(current_chunk, current_start, i-1, file_path, language))
                
                current_chunk = [line]
                current_start = i
                in_component = True
                component_name = match.group(1)
                brace_count = line.count('{') - line.count('}')
                paren_count = line.count('(') - line.count(')')
            else:
                current_chunk.append(line)
                if in_component:
                    brace_count += line.count('{') - line.count('}')
                    paren_count += line.count('(') - line.count(')')
                    
                    # Component ended (simplified logic)
                    if (brace_count <= 0 and paren_count <= 0 and 
                        any(char in ''.join(current_chunk) for char in '{}()')):
                        chunks.append(self._create_react_chunk(current_chunk, current_start, i, file_path, language, component_name))
                        current_chunk = []
                        in_component = False
                        brace_count = 0
                        paren_count = 0
        
        if current_chunk:
            chunks.append(self._create_react_chunk(current_chunk, current_start, len(lines)-1, file_path, language))
        
        return self._merge_small_chunks(chunks)
    
    def _chunk_generic(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Generic chunking for unsupported languages"""
        chunks = []
        lines = content.split('\n')
        
        chunk_size = self.max_chunk_size // 4  # Smaller chunks for generic
        overlap = self.overlap_size
        
        current_chunk = []
        current_tokens = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            line_tokens = self._count_tokens(line)
            
            if current_tokens + line_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    chunk_type="code_block",
                    start_line=start_line + 1,
                    end_line=i,
                    language=language
                ))
                
                # Start new chunk with overlap
                overlap_lines = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_lines + [line]
                current_tokens = sum(self._count_tokens(l) for l in current_chunk)
                start_line = i - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(CodeChunk(
                content='\n'.join(current_chunk),
                file_path=file_path,
                chunk_type="code_block",
                start_line=start_line + 1,
                end_line=len(lines),
                language=language
            ))
        
        return chunks
    
    def _create_js_chunk(self, lines: List[str], start: int, end: int, file_path: str, language: str) -> CodeChunk:
        """Create JavaScript chunk"""
        content = '\n'.join(lines)
        
        # Extract function name
        func_match = re.search(r'(?:function\s+(\w+)|const\s+(\w+)\s*=)', content)
        function_name = func_match.group(1) or func_match.group(2) if func_match else None
        
        # Extract class name
        class_match = re.search(r'class\s+(\w+)', content)
        class_name = class_match.group(1) if class_match else None
        
        chunk_type = "class" if class_name else "function" if function_name else "code_block"
        
        return CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type=chunk_type,
            start_line=start + 1,
            end_line=end + 1,
            language=language,
            function_name=function_name,
            class_name=class_name
        )
    
    def _create_react_chunk(self, lines: List[str], start: int, end: int, file_path: str, 
                           language: str, component_name: str = None) -> CodeChunk:
        """Create React component chunk"""
        content = '\n'.join(lines)
        
        if not component_name:
            # Try to extract component name
            comp_match = re.search(r'(?:const\s+|function\s+)([A-Z]\w+)', content)
            component_name = comp_match.group(1) if comp_match else None
        
        return CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type="component",
            start_line=start + 1,
            end_line=end + 1,
            language=language,
            function_name=component_name
        )
    
    def _extract_python_imports_and_globals(self, content: str, file_path: str) -> List[CodeChunk]:
        """Extract imports and global variables as separate chunks"""
        chunks = []
        lines = content.split('\n')
        
        import_lines = []
        global_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ') or
                stripped.startswith('__version__') or stripped.startswith('__author__')):
                import_lines.append((i, line))
            elif (not stripped.startswith('def ') and not stripped.startswith('class ') and
                  '=' in stripped and not stripped.startswith('#')):
                global_lines.append((i, line))
        
        # Create import chunk
        if import_lines:
            import_content = '\n'.join([line for _, line in import_lines])
            chunks.append(CodeChunk(
                content=import_content,
                file_path=file_path,
                chunk_type="import",
                start_line=import_lines[0][0] + 1,
                end_line=import_lines[-1][0] + 1,
                language="Python"
            ))
        
        return chunks
    
    def _calculate_python_complexity(self, node: ast.FunctionDef) -> float:
        """Calculate cyclomatic complexity for Python function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return float(complexity)
    
    def _extract_python_dependencies(self, node: ast.AST) -> List[str]:
        """Extract function calls and attribute access"""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split())  # Fallback
    
    def _merge_small_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Merge small chunks to optimize storage"""
        if not chunks:
            return chunks
        
        merged = []
        current_chunk = None
        
        for chunk in chunks:
            chunk_size = self._count_tokens(chunk.content)
            
            if chunk_size < 100:  # Very small chunk
                if (current_chunk and current_chunk.chunk_type == chunk.chunk_type and
                    self._count_tokens(current_chunk.content + chunk.content) < self.max_chunk_size):
                    # Merge with previous
                    current_chunk.content += "\n\n" + chunk.content
                    current_chunk.end_line = chunk.end_line
                    continue
            
            if current_chunk:
                merged.append(current_chunk)
            current_chunk = chunk
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged

    def _split_large_class(self, node: ast.ClassDef, lines: List[str], file_path: str) -> CodeChunk:
        """Split large class into method chunks"""
        # For now, return the class as-is but could be enhanced to split methods
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if node.end_lineno else start_line + 50
        content = '\n'.join(lines[start_line:end_line])
        
        return CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type="class",
            start_line=start_line + 1,
            end_line=end_line,
            language="Python",
            class_name=node.name
        )


class SemanticCodeAnalyzer:
    """Semantic code analyzer with ChromaDB integration"""
    
    def __init__(self, collection_name: str = "codebase", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunker = CodeChunker()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use OpenAI embeddings
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
    
    def index_codebase(self, files: List[Path], base_path: Path) -> int:
        """Index codebase files into ChromaDB"""
        total_chunks = 0
        
        for file_path in files:
            if file_path.suffix not in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                continue
                
            try:
                content = self._read_file_safely(file_path)
                language = self._get_language(file_path)
                relative_path = str(file_path.relative_to(base_path))
                
                # Chunk the code
                chunks = self.chunker.chunk_code(content, relative_path, language)
                
                if chunks:
                    self._add_chunks_to_collection(chunks)
                    total_chunks += len(chunks)
                    print(f"Indexed {len(chunks)} chunks from {relative_path}")
                
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
        
        return total_chunks
    
    def semantic_search(self, query: str, n_results: int = 10, 
                       filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search on the codebase"""
        where_clause = {}
        
        if filter_params:
            if 'language' in filter_params:
                where_clause['language'] = filter_params['language']
            if 'chunk_type' in filter_params:
                where_clause['chunk_type'] = filter_params['chunk_type']
            if 'file_path' in filter_params:
                where_clause['file_path'] = {"$contains": filter_params['file_path']}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return self._format_search_results(results)
    
    def find_similar_code(self, code_snippet: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar code snippets"""
        return self.semantic_search(code_snippet, n_results)
    
    def get_code_context(self, file_path: str, function_name: str = None) -> List[Dict[str, Any]]:
        """Get code context for a specific file or function"""
        where_clause = {"file_path": {"$contains": file_path}}
        
        if function_name:
            where_clause["function_name"] = function_name
        
        results = self.collection.query(
            query_texts=[f"function {function_name}" if function_name else f"file {file_path}"],
            n_results=20,
            where=where_clause
        )
        
        return self._format_search_results(results)
    
    def answer_code_question(self, question: str, context_size: int = 5) -> Dict[str, Any]:
        """Answer questions about the codebase using RAG"""
        # Get relevant context
        relevant_chunks = self.semantic_search(question, context_size)
        
        if not relevant_chunks:
            return {"answer": "No relevant code found for your question.", "sources": []}
        
        # Prepare context for LLM
        context = self._prepare_rag_context(relevant_chunks)
        
        return {
            "question": question,
            "context": context,
            "sources": [chunk["metadata"]["file_path"] for chunk in relevant_chunks],
            "relevant_chunks": relevant_chunks
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed codebase"""
        try:
            count = self.collection.count()
            
            # Get sample to analyze composition
            sample_results = self.collection.query(
                query_texts=["code"],
                n_results=min(100, count)
            )
            
            if not sample_results['metadatas']:
                return {"total_chunks": 0}
            
            metadatas = sample_results['metadatas'][0]
            
            # Analyze composition
            languages = {}
            chunk_types = {}
            
            for metadata in metadatas:
                lang = metadata.get('language', 'unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                
                languages[lang] = languages.get(lang, 0) + 1
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            return {
                "total_chunks": count,
                "languages": languages,
                "chunk_types": chunk_types,
                "sample_size": len(metadatas)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self):
        """Clear the ChromaDB collection"""
        try:
            self.client.delete_collection(self.collection_name)
            # Recreate empty collection
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=openai_ef
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def _read_file_safely(self, file_path: Path) -> str:
        """Safely read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as f:
                return f.read()
        except Exception:
            return ""
    
    def _get_language(self, file_path: Path) -> str:
        """Get language from file extension"""
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.ts': 'TypeScript',
            '.jsx': 'JSX',
            '.tsx': 'TSX'
        }
        return extension_map.get(file_path.suffix, 'Unknown')
    
    def _add_chunks_to_collection(self, chunks: List[CodeChunk]):
        """Add code chunks to ChromaDB collection"""
        if not chunks:
            return
        
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        ids = [chunk.embedding_id for chunk in chunks]
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
    
    def _format_search_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB search results"""
        if not results['documents'] or not results['documents'][0]:
            return []
        
        formatted_results = []
        
        for i in range(len(results['documents'][0])):
            result = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else None,
                "id": results['ids'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _prepare_rag_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context for RAG queries"""
        context_parts = []
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            file_info = f"File: {metadata['file_path']}"
            
            if metadata.get('function_name'):
                file_info += f", Function: {metadata['function_name']}"
            elif metadata.get('class_name'):
                file_info += f", Class: {metadata['class_name']}"
            
            context_parts.append(f"{file_info}\n{chunk['content']}\n")
        
        return "\n---\n".join(context_parts)