# Medical RAG Chatbot

A **Retrieval-Augmented Generation (RAG) based Medical Chatbot** that answers questions using information retrieved from a medical knowledge base.

This project was built as a practical learning project to understand how modern GenAI applications work — from processing documents and creating embeddings to vector search, LLM integration, backend APIs, and a React frontend.

> **Note:** This project is for educational purposes only and is not a replacement for professional medical advice, diagnosis, or treatment.

---

## Table of Contents

- [Overview](#overview)
- [What is RAG?](#what-is-rag)
- [Project Features](#project-features)
- [How the Project Works](#how-the-project-works)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [Document Ingestion](#document-ingestion)
- [API Endpoints](#api-endpoints)
- [Example Workflow](#example-workflow)
- [Why Pinecone?](#why-pinecone)
- [Why Embeddings?](#why-embeddings)
- [Why RAG Instead of Only an LLM?](#why-rag-instead-of-only-an-llm)
- [Challenges and Learning](#challenges-and-learning)
- [Future Improvements](#future-improvements)
- [Medical Disclaimer](#medical-disclaimer)
- [Learning Journey](#learning-journey)

---

# Overview

The Medical RAG Chatbot allows users to ask questions related to the information available in a medical knowledge base.

Instead of sending a question directly to an LLM and relying only on its pretrained knowledge, this application first searches a vector database for relevant information.

The retrieved information is then provided to the LLM as context so that the response is grounded in the available medical reference material.

### Basic idea

```text
User Question
      ↓
Retrieve Relevant Information
      ↓
Pinecone Vector Search
      ↓
Relevant Medical Context
      ↓
LLM
      ↓
Generated Answer
      ↓
React UI
```

---

# What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

It combines two major ideas:

### Retrieval

The system searches an external knowledge source for information relevant to the user's question.

In this project, the knowledge source is a medical document that has been processed and stored as vectors in **Pinecone**.

### Generation

After retrieving the relevant information, the retrieved context is sent to an LLM.

The LLM uses that context to generate a natural-language answer.

### RAG flow

```text
Medical Document
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embeddings
      ↓
Pinecone
      ↓

                    User Question
                         ↓
                    Query Embedding
                         ↓
                   Similarity Search
                         ↓
                  Relevant Chunks
                         ↓
                 Prompt + Context
                         ↓
                       LLM
                         ↓
                    Final Answer
```

---

# Project Features

## AI / RAG Features

- Retrieval-Augmented Generation
- Semantic document search
- Vector similarity search
- Medical knowledge retrieval
- Context-based LLM responses
- Source/page references
- Structured responses
- Markdown-supported answers

## Chat Features

- Multiple conversations
- New chat functionality
- Persistent chat history
- Regenerate response
- Copy response
- Conversation history stored locally
- Starter questions

## UI Features

- Clean and responsive interface
- Chat-based interaction
- Markdown rendering
- Headings and lists
- Tables
- Code formatting
- Loading animation
- Expandable source references
- Mobile-friendly layout

---

# How the Project Works

The application can be divided into two major parts:

```text
                 MEDICAL RAG SYSTEM
                         │
            ┌────────────┴────────────┐
            │                         │
       DATA PIPELINE             QUERY PIPELINE
            │                         │
       Medical PDF               User Question
            ↓                         ↓
     Text Extraction             Embedding
            ↓                         ↓
      Text Chunking             Pinecone Search
            ↓                         ↓
       Embeddings               Relevant Chunks
            ↓                         ↓
        Pinecone              Context + Question
                                      ↓
                                     LLM
                                      ↓
                                   Answer
```

---

# 1. Document Processing

The first step is to process the medical document.

The PDF is loaded using LangChain's document loaders.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Medical_book.pdf")
documents = loader.load()
```

The document is converted into text that can be processed by the RAG pipeline.

---

# 2. Text Chunking

Large documents cannot always be processed as one large piece of text.

Therefore, the document is divided into smaller chunks.

This project uses:

```python
RecursiveCharacterTextSplitter
```

Example configuration:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)
```

### Why chunking?

Instead of searching through an entire book for every question, the system searches smaller pieces of information.

This makes retrieval more efficient and helps the LLM receive more relevant context.

---

# 3. Creating Embeddings

After splitting the document, each chunk is converted into a numerical representation called an **embedding**.

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through Hugging Face embeddings.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The embedding model converts text into a vector.

For this model, the vector dimension is:

```text
384
```

---

# 4. Storing Vectors in Pinecone

The generated vectors are stored in **Pinecone**, a vector database.

The project uses a Pinecone index named:

```text
medical-chatbot
```

The index uses:

```text
Metric: Cosine Similarity
Dimension: 384
```

The vectors are stored along with their original text and metadata.

This allows the application to find the document chunks that are semantically similar to a user's question.

---

# 5. User Asks a Question

For example:

```text
What is acne?
```

The question is converted into an embedding using the same embedding model.

```text
"What is acne?"
       ↓
Embedding
       ↓
384-dimensional vector
```

---

# 6. Similarity Search

The query vector is compared with vectors stored in Pinecone.

The system retrieves the most relevant chunks.

The application uses multiple retrieved chunks instead of relying on a single result.

Example:

```text
User Question
      ↓
Pinecone
      ↓
Top relevant chunks
      ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
Chunk 5
Chunk 6
```

These chunks become the context for the LLM.

---

# 7. Context + Prompt

The retrieved information is added to the prompt sent to the LLM.

Conceptually:

```text
System Instructions
        +
Retrieved Medical Context
        +
User Question
        ↓
       LLM
        ↓
     Answer
```

The model is instructed to use the retrieved information when answering.

If enough information cannot be found in the available reference, the application is designed to avoid simply inventing an answer.

---

# 8. LLM Response

The project uses Groq with:

```text
openai/gpt-oss-20b
```

through LangChain.

Example:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1
)
```

The retrieved context and user's question are passed to the model.

The generated answer is then returned by the Flask backend.

---

# 9. React Frontend

The frontend is built using:

- React
- Vite
- Tailwind CSS
- React Markdown
- Lucide React

The frontend communicates with the Flask backend using HTTP requests.

```text
React
  ↓
POST /api/chat
  ↓
Flask
  ↓
RAG Pipeline
  ↓
Groq + Pinecone
  ↓
Flask Response
  ↓
React
  ↓
Display Answer
```

---

# Architecture

The overall architecture looks like this:

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ↓
                         ┌──────────────────┐
                         │  React Frontend  │
                         │  Tailwind + Vite │
                         └────────┬─────────┘
                                  │
                             HTTP Request
                                  │
                                  ↓
                         ┌──────────────────┐
                         │ Flask Backend    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ↓                           ↓
            ┌───────────────┐           ┌───────────────┐
            │   Pinecone    │           │     Groq      │
            │ Vector Search │           │      LLM      │
            └───────┬───────┘           └───────┬───────┘
                    │                           │
                    │       Retrieved Context   │
                    └─────────────┬─────────────┘
                                  ↓
                           Generated Answer
                                  │
                                  ↓
                         ┌──────────────────┐
                         │  React Frontend  │
                         └──────────────────┘
```

---

# Technology Stack

## Frontend

| Technology | Purpose |
|---|---|
| React.js | Building the user interface |
| Vite | Frontend development/build tool |
| Tailwind CSS | Styling |
| React Markdown | Rendering Markdown responses |
| Lucide React | UI icons |

## Backend

| Technology | Purpose |
|---|---|
| Python | Main backend language |
| Flask | REST API |
| Flask-CORS | Frontend/backend communication |

## AI / RAG

| Technology | Purpose |
|---|---|
| LangChain | RAG pipeline and LLM integration |
| Groq | LLM inference |
| `openai/gpt-oss-20b` | Language model |
| Hugging Face | Embedding model |
| `all-MiniLM-L6-v2` | Text embeddings |

## Database

| Technology | Purpose |
|---|---|
| Pinecone | Vector database |
| Cosine Similarity | Vector similarity metric |

---

# Project Structure

```text
medical-rag-chatbot/
│
├── backend/
│   │
│   ├── app.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
├── frontend/
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   │
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── .gitignore
└── README.md
```

---

# Prerequisites

Before running the project, install:

### Python

Recommended version:

```text
Python 3.12
```

### Node.js

Install Node.js and npm.

Check installation:

```bash
node --version
npm --version
```

### Pinecone

You need a Pinecone account and API key.

### Groq

You need a Groq API key to access the LLM.

---

# Installation

## Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-chatbot.git
```

Move into the project:

```bash
cd medical-rag-chatbot
```

---

# Step 2 — Create Python Virtual Environment

For Windows:

```bash
py -3.12 -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

You should see something similar to:

```text
(.venv)
```

at the beginning of your terminal.

---

# Step 3 — Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

---

# Step 4 — Configure Environment Variables

Inside the `backend` directory, create:

```text
.env
```

Add:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_INDEX_NAME=medical-chatbot
```

### Important

Never commit your `.env` file to GitHub.

Your `.gitignore` should contain:

```text
.env
.venv/
__pycache__/
node_modules/
dist/
```

---

# Step 5 — Start Flask Backend

From the project directory:

```bash
cd backend
python app.py
```

The backend should run at:

```text
http://localhost:5000
```

You can check whether it is running using:

```text
http://localhost:5000/api/health
```

---

# Step 6 — Install Frontend Dependencies

Open another terminal.

Move to the frontend:

```bash
cd frontend
```

Install packages:

```bash
npm install
```

---

# Step 7 — Start Frontend

Run:

```bash
npm run dev
```

Vite will provide a local address, normally:

```text
http://localhost:5173
```

Open the address in your browser.

---

# Environment Variables

The project uses environment variables to keep API keys outside the source code.

Example:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_INDEX_NAME=medical-chatbot
```

### Why use `.env`?

API keys are private credentials.

They should not be written directly inside Python files or uploaded to GitHub.

Bad:

```python
GROQ_API_KEY = "my-secret-api-key"
```

Better:

```python
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

---

# Document Ingestion

The medical PDF is required when creating the vector database.

The ingestion process is:

```text
Medical_book.pdf
       ↓
PyPDFLoader
       ↓
Extracted Documents
       ↓
RecursiveCharacterTextSplitter
       ↓
Text Chunks
       ↓
Hugging Face Embeddings
       ↓
Pinecone
```

Once the document has been embedded and stored in Pinecone, the chatbot can retrieve the stored information without loading the PDF every time the Flask application starts.

### If the PDF changes

If you replace or update the medical PDF, the new document needs to be processed and indexed again.

Otherwise, Pinecone will continue to contain the previously indexed information.

---

# API Endpoints

## Health Check

### Request

```http
GET /api/health
```

### Purpose

Checks whether the Flask backend is running correctly.

Example response:

```json
{
  "status": "ok"
}
```

---

# Chat Endpoint

### Request

```http
POST /api/chat
```

Example:

```json
{
  "message": "What is acne?",
  "history": []
}
```

### Response

```json
{
  "answer": "Acne is a skin condition...",
  "sources": [
    {
      "page": "Page 12"
    }
  ]
}
```

---

# Regenerate Endpoint

### Request

```http
POST /api/regenerate
```

This endpoint allows the frontend to request another response for the same question.

Example:

```json
{
  "message": "What is acne?",
  "history": []
}
```

---

# Example Workflow

Suppose the user asks:

```text
What is acne?
```

The application performs the following steps:

### Step 1

React sends the question to Flask.

```text
"What is acne?"
```

### Step 2

Flask sends the query through the RAG pipeline.

### Step 3

The query is converted into an embedding.

### Step 4

Pinecone searches for semantically similar medical content.

### Step 5

The most relevant document chunks are retrieved.

### Step 6

The retrieved chunks are added to the prompt.

### Step 7

The prompt is sent to the Groq LLM.

### Step 8

The LLM generates the response.

### Step 9

Flask returns the answer and source information.

### Step 10

React displays the response to the user.

---

# Why Pinecone?

Traditional databases are good at searching structured information.

Vector databases are designed for searching based on **semantic similarity**.

For example:

```text
Question:
"Why do people get pimples?"
```

The document might contain:

```text
"Acne is a common inflammatory condition
affecting the pilosebaceous unit..."
```

The wording is different, but the meaning is related.

Embeddings allow the system to identify this semantic relationship.

---

# Why Embeddings?

An embedding converts text into numbers that represent the meaning of the text.

For example:

```text
"Acne is a skin condition"
            ↓
       Embedding Model
            ↓
[0.12, -0.43, 0.81, ...]
```

The exact numbers are not important to the user.

What matters is that similar pieces of text tend to have similar vector representations.

This allows Pinecone to perform semantic search.

---

# Why RAG Instead of Only an LLM?

A normal LLM can answer questions using the knowledge it learned during training.

However, an application may need to answer using a specific document or knowledge base.

RAG adds an external retrieval step.

### Without RAG

```text
Question
   ↓
LLM
   ↓
Answer
```

### With RAG

```text
Question
   ↓
Search Knowledge Base
   ↓
Relevant Information
   ↓
LLM + Context
   ↓
Answer
```

This makes it possible to build applications around specific private or domain-specific knowledge.

---

# Challenges I Faced

Building this project also helped me understand some of the practical challenges involved in GenAI applications.

### 1. Understanding the RAG pipeline

Initially, the different components such as embeddings, vector databases, retrievers, prompts, and LLMs can feel disconnected.

Building the complete pipeline helped me understand how they work together.

### 2. Retrieval quality

The quality of the final answer depends heavily on the quality of the retrieved context.

If the wrong chunks are retrieved, even a powerful LLM may produce a poor answer.

### 3. Prompt design

The prompt needs to clearly tell the model how to use the retrieved information.

### 4. Connecting frontend and backend

The project also required connecting a React frontend with a Python Flask API.

This helped me understand how an AI backend can be integrated into a real web application.

---

# What I Learned

Through this project, I got hands-on experience with:

- Generative AI
- Retrieval-Augmented Generation
- LangChain
- Document processing
- PDF text extraction
- Text chunking
- Embeddings
- Semantic search
- Vector databases
- Pinecone
- Prompt engineering
- LLM integration
- Groq
- Flask APIs
- React
- Tailwind CSS
- Frontend/backend communication
- Environment variables
- Python virtual environments

Most importantly, I learned that building a GenAI application involves much more than simply calling an LLM API.

There are multiple components working together:

```text
Data
 ↓
Processing
 ↓
Embeddings
 ↓
Vector Database
 ↓
Retrieval
 ↓
Prompt
 ↓
LLM
 ↓
Application
```

---

# Future Improvements

There are several things I would like to improve in the future.

### RAG Improvements

- Improve chunking strategy
- Experiment with different embedding models
- Add reranking
- Improve retrieval accuracy
- Add hybrid search
- Evaluate retrieval quality

### Application Improvements

- Allow users to upload their own documents
- Support multiple knowledge bases
- Add user authentication
- Add conversation persistence using a database
- Add streaming LLM responses
- Add better source citations
- Add document management

### Deployment

The application can also be extended for deployment using services such as:

```text
Frontend → Vercel / Netlify
Backend  → Render / Railway / Cloud
Database → Pinecone
```

---

# Security Considerations

API keys should never be exposed in the frontend.

Keep sensitive credentials inside environment variables.

Never commit:

```text
.env
```

to GitHub.

Also, when deploying this application, make sure:

- CORS is configured correctly
- API keys are stored securely
- User input is validated
- Rate limiting is considered
- Sensitive medical information is handled carefully

---

# Medical Disclaimer

This chatbot is an **educational project** created to demonstrate Retrieval-Augmented Generation.

It should **not** be used for:

- Medical diagnosis
- Emergency decisions
- Prescribing medication
- Replacing a doctor
- Making treatment decisions

Always consult a qualified healthcare professional for medical advice.

---

# Learning Journey

I built this project while learning Generative AI through the **KRIVA GenAI Workshop**.

The workshop gave me the opportunity to move from learning individual concepts to actually building an end-to-end GenAI application.

This project helped me connect concepts that I had previously studied separately and understand how they work together in a real application.

I'm still learning and experimenting, and I plan to keep building more projects around **GenAI, RAG, LLMs, and AI applications**.

---

# Author

**Sarthak Agarwal**

Student | Learning Generative AI & Software Development

---

## ⭐ If you found this project useful

Feel free to explore the repository, try the project, or use the ideas to build your own RAG application.

**Built while learning GenAI, one project at a time.**
