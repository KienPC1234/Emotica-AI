Emotica AI Server (LLaMA + Vision + RAG)

This is a high-performance FastAPI server designed to integrate multiple AI functionalities into a single, OpenAI-compatible API:

LLaMA (Text): Serves GGUF models using llama-cpp-python for fast, GPU-accelerated text generation.

Vision (Captioning): Pre-processes multi-modal inputs, generating text descriptions for images (using image2text) before sending them to the LLaMA model.

RAG (Retrieval-Augmented Generation): Injects relevant context from a vector database (rag_manager) into the prompt to provide grounded, accurate answers.

Features

Dynamic Model Management: Switch LLaMA models and Vision models via API endpoints without restarting the server.

Configuration Reloading: Reload model configurations (cfg.json) on the fly.

RAG Database Management: Upload new documents (.pdf, .txt, .md), list existing documents, and delete documents from the vector store.

RAG Rebuild: Force a full rebuild of the vector database on startup using a command-line flag.

System Monitoring: GET /system/stats endpoint to monitor VRAM (NVIDIA) or CPU/RAM usage.

OpenAI-Compatible Endpoint: The core /v1/chat/completions endpoint handles text, images, and RAG seamlessly.

Optimized Locking: Efficient locking ensures that RAG and Vision preprocessing do not block the LLaMA model, maximizing throughput.

1. Setup

Installation

pip install -r requirements.txt

Install llama-cpp-python with hardware acceleration (cuBLAS for NVIDIA recommended):

# Example for NVIDIA:
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
# For other options (Metal, CLBlast), see llama-cpp-python docs.


Model Configuration (models/cfg.json)

The server loads LLaMA models based on the models/cfg.json file. You must create this file and point it to your GGUF models.

Example models/cfg.json:

{
  "models": [
    {
      "name": "Llama-3-8B-Instruct",
      "path": "D:/models/llama-3-8b-instruct.Q5_K_M.gguf",
      "format": "llama-3",
      "description": "Meta's Llama 3 8B Instruct (Quantized)",
      "max_context": 8192,
      "temperature": 0.7,
      "top_p": 0.9
    },
    {
      "name": "Mistral-7B-Instruct-v0.2",
      "path": "D:/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
      "format": "mistral-instruct",
      "description": "Mistral 7B Instruct v0.2",
      "max_context": 4096,
      "temperature": 0.7,
      "top_p": 0.9
    }
  ],
  "valid_formats": ["llama-3", "mistral-instruct", "chatml"]
}


2. Running the Server

Standard Run

The server will load the first model listed in cfg.json as the default.

python llama_server.py


The server will be available at http://127.0.0.1:8002 (by default, or as configured in ports.toml).
API docs are available at http://127.0.0.1:8002/docs.

Rebuilding the RAG Database (New)

To force a full rebuild of your vector database (e.g., after adding many files to the source directory), use the --rebuild-rag flag.

Note: You must implement the rebuild_database() logic within rag_manager.py for this flag to work.

python llama_server.py --rebuild-rag


3. API Endpoints

Core Inference

POST /v1/chat/completions

This is the main endpoint for all interactions. It's compatible with the OpenAI chat completions format.

Request Body:

messages (List): The conversation history.

stream (bool, optional): true for a streaming response, false for a single JSON object.

use_rag (bool, optional): true to enable RAG for this request. The server will search the vector DB based on the last user message.

max_tokens, temperature, top_p (optional): Override model defaults.

Example 1: Standard Text Request (with RAG)

{
  "messages": [
    {
      "role": "user",
      "content": "What is the new policy on remote work?"
    }
  ],
  "use_rag": true,
  "stream": false
}


Example 2: Multi-modal (Vision) Request

To send an image, use the multi-part content format. The image must be a data:image/...;base64,... string.

{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is happening in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...[your_base64_image_data].../Z"
          }
        }
      ]
    }
  ],
  "use_rag": false,
  "stream": false
}


LLaMA (Text) Management

GET /models: Lists all available LLaMA models from cfg.json.

GET /models/current: Shows the currently loaded LLaMA model.

POST /models/switch: Switches the active LLaMA model.

Body: {"model_name": "Llama-3-8B-Instruct"}

POST /models/reload-config: Reloads cfg.json from disk.

RAG (Vector DB) Management

POST /rag/upload: Uploads a .pdf, .txt, or .md file to be indexed in the vector database.

Format: multipart/form-data with a file field.

GET /rag/list: Lists the source documents currently in the database.

DELETE /rag/delete: Deletes a document and its vectors from the database.

Query Param: ?filename=example.pdf

Captioner (Vision) Management

GET /captioner/models: Lists supported vision (captioning) models.

GET /captioner/current: Shows the currently loaded vision model.

POST /captioner/switch: Switches the active vision model.

Body: {"model_id": "Salesforce/blip-image-captioning-large"}

System & Monitoring

GET /system/stats: Returns a JSON object with VRAM or CPU/RAM usage statistics.