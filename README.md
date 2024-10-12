
# RAG Application with Llama 3.1 and Sentence Transformers

## Reference

This project is inspired by the tutorial on DataCamp: [How to Use Llama 3.1 for Retrieval-Augmented Generation (RAG)](https://www.datacamp.com/tutorial/llama-3-1-rag).
This project demonstrates how to build a Retrieval-Augmented Generation (RAG) application using Llama 3.1 and Sentence Transformers. The application retrieves relevant documents from a set of URLs and generates concise answers to user queries.

## Files

- `rag_application.py`: The main application file that initializes the retriever and RAG chain, and processes user queries.
- `doc_store.py`: Contains functions to create a retriever and RAG chain using Sentence Transformers and Llama 3.1.

## Setup

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2. **Create and activate a virtual environment**:
    - On Windows:
      ```sh
      python -m venv venv
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application**:
    ```sh
    python rag_application.py
    ```

2. **Example output**:
    ```
    Question: What is prompt engineering?
    Answer: Prompt engineering is the process of designing and refining prompts to effectively communicate with language models and achieve desired outputs.
    ```

## Code Overview

### `doc_store.py`

This file contains the following functions:

- `get_retriever()`: Creates a retriever using Sentence Transformers to embed documents and store them in an SKLearn vector store.
- `get_rag_chain()`: Creates a RAG chain using a prompt template and the Llama 3.1 language model.

### `rag_application.py`

This file contains the main application logic:

- Initializes the retriever and RAG chain using functions from `doc_store.py`.
- Processes user queries by retrieving relevant documents and generating answers using the RAG chain.
