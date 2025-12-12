# Voice-to-Voice Product Recommendation System

A conversational voice-based product recommendation system powered by an agentic RAG pipeline.

## Deployment
1. Reinstall environement. Run
`uv venv .venv`

`uv sync`

uv library should automatically reinstall all the libraries we used by reading the `uv.lock` and `pyproject.toml` files.

2. Run `uv run src/store_chromadb.py` to build chroma database for the amazon data.
3. Run `uv run streamlit run streamlit_app.py` to start the streamlit app.


## Overview

This Streamlit application enables users to:
1. **Record voice queries** for product recommendations
2. **Process queries** through an intelligent agentic pipeline
3. **Receive voice responses** with product recommendations and comparisons

## Pipeline Architecture

The system uses a **4-stage agentic RAG pipeline** built with LangGraph:

### Stage 1: **Router Node**
- **Purpose**: Classify user intent and extract query details
- **Functionality**:
  - Determines route type: `search` (product lookup), `general` (conversational), or `unsafe` (flagged content)
  - Extracts task description and constraints (budget, brand, category, etc.)
  - Flags any safety concerns
  - Cleans and normalizes the input query

### Stage 2: **Planner Node**
- **Purpose**: Create a retrieval strategy
- **Functionality**:
  - Decides which data sources to query:
    - **Private**: Internal product catalog (structured, local)
    - **Live**: Web search for real-time data (current prices, reviews, deals)
  - Specifies which fields to retrieve (title, price, brand, ingredients, etc.)
  - Defines comparison criteria for ranking products
  - Refines search parameters and filters

### Stage 3: **Retriever Node**
- **Purpose**: Fetch relevant products
- **Functionality**:
  - Calls the RAG MCP (Model Context Protocol) server
  - Searches the Chroma vector database for similar products
  - Applies filters (price range, brand, category, etc.)
  - Re-ranks results by relevance
  - Returns structured product data with unique identifiers

### Stage 4: **Answerer Node**
- **Purpose**: Generate final recommendations
- **Functionality**:
  - Synthesizes results into a natural, concise recommendation
  - Ensures all claims are grounded in retrieval results
  - Includes proper citations with product IDs and web links
  - Checks for safety flags before responding
  - Formats recommendations for voice conversion

## Streamlit App Flow

1. **Record Query**: User records a voice message with their product request
2. **Speech-to-Text**: Audio is converted to text using OpenAI's Whisper API
3. **Pipeline Processing**: Query flows through the 4-stage pipeline:
   - Router analyzes intent
   - Planner creates retrieval strategy
   - Retriever fetches relevant products
   - Answerer generates recommendation
4. **Results Display**: Shows:
   - Transcribed query
   - Final recommendation text
   - Product comparison table
   - Related products
5. **Text-to-Speech**: Response is converted back to audio for playback

## Key Features

- **Agentic RAG**: Intelligent routing and planning before retrieval
- **Hybrid Search**: Combines private product catalog + web search
- **Safety Filtering**: Detects and blocks inappropriate requests
- **Citations & Grounding**: All recommendations are sourced with IDs and links
- **Voice Interface**: Fully conversational voice input/output
- **Product Comparison**: Displays side-by-side product comparisons

## Data Sources

- **Private Catalog**: Chroma vector database (`chroma_amazon_clean/`) with Amazon product data
- **Product Fields**: title, brand, category, price, ingredients, description, unique identifiers
- **Live Search**: Optional web search for real-time availability and deals

## Example Queries

- "I want to buy Barbie dolls."
- "What is Catan the board game?"
- "I want to buy a skateboard under 300 dollars."
- "Find eco-friendly cleaners under $20"

## Configuration

The system uses prompts defined in the `data/` directory:
- `prompt_router.txt`: Router classification logic
- `prompt_planner.txt`: Retrieval strategy planning
- `prompt_retriever.txt`: Retrieval execution
- `prompt_answerer.txt`: Response generation

---

**Built with**: LangGraph, OpenAI, Streamlit, Chroma DB, FastMCP
