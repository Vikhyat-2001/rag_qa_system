# Question-Answering System with RAG (Retrieval-Augmented Generation)

A question-answering system that retrieves relevant information from a knowledge base and generates answers using a pre-trained language model (LLM). Built with **Python**, **FAISS**, **Sentence Transformers**, and **Flask**.

---

## Features

- **Retrieval Mechanism**:
  - Uses **FAISS** for efficient document retrieval.
  - Embeds documents and queries using **Sentence Transformers**.

- **Answer Generation**:
  - Uses a pre-trained language model (e.g., GPT-2) to generate answers.

- **REST API**:
  - Exposes the QA system as a REST API using **Flask**.

---

## Technologies Used

- **Retrieval**: FAISS, Sentence Transformers
- **Generation**: Hugging Face Transformers (GPT-2)
- **API**: Flask
- **Containerization**: Docker (optional)

---

## Setup Instructions

### Prerequisites

1. Install [Python 3.9](https://www.python.org/downloads/).
2. Install [Docker](https://docs.docker.com/get-docker/) (optional).

---

### Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/rag-qa-system.git
   cd rag-qa-system