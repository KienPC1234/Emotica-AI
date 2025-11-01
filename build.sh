#!/bin/bash

# --- Linux Build Script for Multi-Process Application (v5) ---
# Changes:
# - Removed all --enable-plugin flags as requested.
# - Tries to build with static libpython first, falls back to non-static if it fails.
# - Downloads two specified GGUF models into the build directory.
# - Manually copies the tts_models directory.

echo "--- Starting Advanced Linux Build (v5) ---"

# --- Configuration ---
FINAL_DIR="build_output"
LLAMA_EXEC_NAME="llama_server_exec"
WEB_EXEC_NAME="web_server_exec"
MANAGER_EXEC_NAME="start_app"

URL_LLAMA2="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"
URL_VISTRAL="https://huggingface.co/janhq/Vistral-7b-Chat-GGUF/resolve/main/vitral-7b-chat.Q4_K_M.gguf?download=true"

# --- Build Steps ---

echo "[1/9] Cleaning up previous build directory..."
rm -rf "$FINAL_DIR"
mkdir -p "$FINAL_DIR"

# --- Data File Exclusion ---
# Note: tts_models is now copied manually, so it's removed from this list.
NUITKA_DATA_EXCLUDES="--noinclude-data-files=models/** --noinclude-data-files=vectordb/** --noinclude-data-files=embedding_models/** --noinclude-data-files=image_captioners/** --noinclude-data-files=logs/** --noinclude-data-files=temp_uploads/**"

# --- Nuitka Compilation Function with Fallback ---
compile_with_fallback() {
    local component_name=$1
    shift
    local nuitka_args=($@)

    echo "--> Attempting to compile '$component_name' with static libpython..."
    # Try with static libpython first
    python3 -m nuitka --static-libpython=yes "${nuitka_args[@]}"
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "--> Static build failed. Retrying '$component_name' without static libpython..."
        # If it fails, try without static libpython
        python3 -m nuitka --static-libpython=no "${nuitka_args[@]}"
        exit_code=$?
    fi

    if [ $exit_code -ne 0 ]; then
        echo "FATAL: Compilation of '$component_name' failed completely!"
        exit 1
    fi
    echo "--> Successfully compiled '$component_name'."
}

# 2. Compile llama_server.py
echo "[2/9] Compiling Llama Server..."
compile_with_fallback "Llama Server" \
    --main=llama_server.py \
    --output-dir="$FINAL_DIR" \
    --output-filename="$LLAMA_EXEC_NAME" \
    $NUITKA_DATA_EXCLUDES \
    --assume-yes-for-downloads --remove-output

# 3. Compile web/main.py
echo "[3/9] Compiling Web Server..."
compile_with_fallback "Web Server" \
    --main=web/main.py \
    --output-dir="$FINAL_DIR" \
    --output-filename="$WEB_EXEC_NAME" \
    $NUITKA_DATA_EXCLUDES \
    --include-data-dir=web/templates=web/templates \
    --include-data-dir=web/statics=web/statics \
    --assume-yes-for-downloads --remove-output

# 4. Compile main.py (Process Manager)
echo "[4/9] Compiling Process Manager..."
compile_with_fallback "Process Manager" \
    --main=main.py \
    --output-dir="$FINAL_DIR" \
    --output-filename="$MANAGER_EXEC_NAME" \
    --assume-yes-for-downloads --remove-output

# 5. Create the modified multiprocess.json
echo "[5/9] Creating modified process configuration..."
cat > "$FINAL_DIR/multiprocess.json" << EOL
{
  "llama_server": [
    "./$LLAMA_EXEC_NAME"
  ],
  "web_server": [
    "./$WEB_EXEC_NAME"
  ]
}
EOL

# 6. Download GGUF Models
echo "[6/9] Downloading GGUF models (this may take a while)..."
mkdir -p "$FINAL_DIR/models"
echo "--> Downloading Llama-2-7B-Chat..."
curl -L "$URL_LLAMA2" -o "$FINAL_DIR/models/llama-2-7b-chat.Q4_K_M.gguf" || { echo "Failed to download Llama-2 model!"; exit 1; }
echo "--> Downloading Vistral-7B-Chat..."
curl -L "$URL_VISTRAL" -o "$FINAL_DIR/models/vitral-7b-chat.Q4_K_M.gguf" || { echo "Failed to download Vistral model!"; exit 1; }

# 7. Copy necessary config files
echo "[7/9] Copying configuration files..."
cp ports.toml "$FINAL_DIR/"
if [ -f "models/cfg.json" ]; then
    cp models/cfg.json "$FINAL_DIR/models/"
fi

# 8. Manually copy TTS models directory
echo "[8/9] Copying TTS models..."
mkdir -p "$FINAL_DIR/web/ai_module"
cp -r web/ai_module/tts_models "$FINAL_DIR/web/ai_module/" || { echo "Failed to copy tts_models directory!"; exit 1; }

# 9. Create empty directories that the app expects to exist at runtime
echo "[9/9] Creating runtime directories..."
mkdir -p "$FINAL_DIR/temp_uploads"
mkdir -p "$FINAL_DIR/logs"
mkdir -p "$FINAL_DIR/vectordb/rawcontents"
mkdir -p "$FINAL_DIR/web/temp_user_image"

echo "--- Build Finished Successfully! ---"
echo "The complete application is in the '$FINAL_DIR' directory."
echo "To run, navigate into the directory and run the start_app executable:"
echo "cd $FINAL_DIR && ./$MANAGER_EXEC_NAME"
