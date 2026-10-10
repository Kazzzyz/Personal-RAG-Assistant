# Personal RAG Assistant

A Retrieval-Augmented Generation (RAG) system that builds an intelligent knowledge base from a person's professional information and answers questions using retrieved, source-grounded context.

The system can combine information from multiple sources such as resumes, LinkedIn profiles, personal websites, GitHub repositories, projects, certifications, and other professional documents.

## Features

- Load professional information from multiple sources
- Extract text from PDF resumes and LinkedIn exports
- Retrieve portfolio and website content
- Automatically retrieve public GitHub repositories and their README files
- Preserve source metadata for retrieved information
- Split documents into manageable text chunks
- Generate semantic embeddings using Sentence Transformers
- Store embeddings in a FAISS vector database
- Perform semantic similarity search
- Retrieve the most relevant information for a user question
- Support source-aware answers to improve transparency and reduce hallucinations

## RAG Pipeline

```text
CV / Resume
LinkedIn
GitHub
Portfolio Website
Certificates
Other Professional Documents
        │
        ▼
   Document Loading
        │
        ▼
     Text Cleaning
        │
        ▼
      Chunking
        │
        ▼
     Embeddings
        │
        ▼
   FAISS Vector DB
        │
        ▼
 Semantic Retrieval
        │
        ▼
Relevant Context + User Question
        │
        ▼
        LLM
        │
        ▼
 Grounded Answer + Sources
```

## Data Sources

### CV / Resume

PDF documents are loaded using `PyPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("resume.pdf")
cv_docs = loader.load()
```

### LinkedIn

LinkedIn profile information can be supplied as an exported PDF and processed using the same PDF-loading pipeline.

### Personal Website

Portfolio and personal website content is retrieved using `WebBaseLoader`.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
website_docs = loader.load()
```

### GitHub

The GitHub REST API is used to retrieve a user's public repositories.

For each repository, the system can collect:

- Repository name
- Description
- README
- Repository URL

Each repository is converted into a LangChain `Document` with its own metadata.

## Document Processing

All loaded documents are combined into a single collection while preserving their individual metadata.

```python
all_docs = (
    cv_docs
    + github_docs
    + linkedin_docs
    + website_docs
)
```

The documents are then divided into smaller chunks:

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(all_docs)
```

## Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
```

## Vector Database

FAISS is used to store and search document embeddings.

```python
from langchain_community.vectorstores import FAISS

vectorDB = FAISS.from_documents(
    chunks,
    embeddings
)
```

The vector database can also be saved locally:

```python
vectorDB.save_local("personal_rag_faiss")
```

## Semantic Retrieval

When a user asks a question, the system searches for the most semantically relevant chunks.

```python
query = "What projects has this person worked on?"

similar_docs = vectorDB.similarity_search(
    query,
    k=3
)
```

## Source Tracking

Each document keeps metadata describing where its information originated, such as:

```text
CV — Page 1
GitHub — Repository Name
Portfolio — Website URL
LinkedIn — Profile PDF
```

This makes it possible to provide more transparent and verifiable RAG responses.

## Technologies

- Python
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- PyPDF
- GitHub REST API
- BeautifulSoup
- Requests
- Jupyter Notebook

## Installation

```bash
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-huggingface
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
pip install beautifulsoup4
pip install requests
```

## Example Questions

- What projects has this person worked on?
- What machine-learning experience does this person have?
- Which programming languages does this person know?
- What technologies were used in their projects?
- What is their educational background?
- What certifications have they completed?
- Which GitHub projects demonstrate NLP experience?

## Goal

The goal of this project is to create a unified professional knowledge base from information distributed across multiple sources.

Instead of searching through a resume, GitHub repositories, portfolio pages, and professional profiles separately, the RAG system retrieves the most relevant information and provides a context-aware answer grounded in the collected sources.
