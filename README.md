# Emotica AI

Emotica AI is a compassionate and therapeutic virtual assistant designed to provide empathetic and supportive conversations. It integrates a local LLaMA model for text generation, a vision model for image captioning, a RAG system for information retrieval, and emotion detection to tailor its responses.

## Features

-   **Local AI:** Runs entirely on your local machine.
-   **Multi-modal:** Understands both text and images.
-   **RAG (Retrieval-Augmented Generation):** Can answer questions based on documents you provide.
-   **Emotion-Aware:** Detects user emotions from images to provide more empathetic responses.
-   **Web Interface:** Easy-to-use chat interface accessible from your browser.
-   **Memory:** Remembers key information shared by the user for a personalized experience.

## Requirements

-   **Python:** Version 3.10, 3.11, or 3.12.
-   **Git:** For cloning the repository.
-   **C++ Compiler:** Required for `llama-cpp-python`.
    -   **Windows:** Install Visual Studio with the "Desktop development with C++" workload.
    -   **macOS:** Install Xcode Command Line Tools (`xcode-select --install`).
    -   **Linux:** Install `build-essential` (`sudo apt-get install build-essential`).
-   **CUDA Toolkit (Optional, for NVIDIA GPUs):** For GPU acceleration. Make sure to install a version compatible with your driver and PyTorch.

## Deployment Guide

### 1. Clone the Repository

```bash
git clone https://github.com/KienPC1234/Emotica-AI.git
cd Emotica-AI
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

-   **Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

-   **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

The application will attempt to install required packages automatically when you run it. However, you can also install them manually.

There are two requirements files:
-   `requirements.txt`: For CPU-based inference.
-   `requirements_cuda.txt`: For NVIDIA GPU-based inference (includes CUDA-enabled PyTorch and `llama-cpp-python`).

**For CPU:**
```bash
pip install -r requirements.txt
```

**For NVIDIA GPU (CUDA):**
```bash
pip install -r requirements_cuda.txt
```
*Note: The automatic installer in `main.py` uses `requirements.txt` by default. You may need to edit `main.py` if you want it to use the CUDA requirements file.*

### 4. Configure Models

1.  Download your desired GGUF-format LLaMA models.
2.  Place them in the `models/` directory.
3.  Edit `models/cfg.json` to configure your models. A sample file is created on first run if it doesn't exist. Make sure the `path` for each model is correct.

    ```json
    {
      "models": [
        {
          "name": "Llama-3-8B-Instruct-Demo",
          "path": "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
          "format": "llama-3",
          "description": "Model LLaMA 3 8B",
          "max_context": 8192,
          "temperature": 0.7,
          "top_p": 0.9
        }
      ],
      "valid_formats": ["llama-3", "mistral-instruct", "chatml"]
    }
    ```

### 5. Run the Application

Simply run the `main.py` script from the project root directory.

```bash
python main.py
```

The script will:
1.  Check if your Python version is compatible (3.10-3.12).
2.  Check for and install any missing dependencies from `requirements.txt`.
3.  Start the LLaMA API server and the Web server.
4.  Automatically open the web interface in your default browser.

The application will be available at `http://localhost:8000` (or the port configured in `ports.toml`).

### 6. Shutting Down

Press `Ctrl+C` in the terminal where `main.py` is running to gracefully shut down all server processes.
