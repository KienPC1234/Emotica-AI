#!/bin/bash

# --- Linux Build Script for Multi-Process Application (v5) ---
# Changes:
# - Removed all --enable-plugin flags as requested.
# - Tries to build with static libpython first, falls back to non-static if it fails.
# - Model download is now handled by the main application.
# - Manually copies the tts_models directory.

echo "--- Starting Advanced Linux Build (v5) ---"

# --- Configuration ---
FINAL_DIR="build_output"
LLAMA_EXEC_NAME="llama_server_exec"
WEB_EXEC_NAME="web_server_exec"
MANAGER_EXEC_NAME="start_app"

# --- Build Steps ---

echo "[1/8] Cleaning up previous build directory..."
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
echo "[2/8] Compiling Llama Server..."
compile_with_fallback "Llama Server" \
    --main=llama_server.py \
    --output-dir="$FINAL_DIR" \
    --output-filename="$LLAMA_EXEC_NAME" \
    $NUITKA_DATA_EXCLUDES \
    --assume-yes-for-downloads --remove-output

# 3. Compile web/main.py
echo "[3/8] Compiling Web Server..."
compile_with_fallback "Web Server" \
    --main=web/main.py \
    --output-dir="$FINAL_DIR" \
    --output-filename="$WEB_EXEC_NAME" \
    $NUITKA_DATA_EXCLUDES \
    --include-data-dir=web/templates=web/templates \
    --include-data-dir=web/statics=web/statics \
    --assume-yes-for-downloads --remove-output

# 4. Compile main.py (Process Manager)
echo "[4/8] Compiling Process Manager..."
compile_with_fallback "Process Manager" \
    --main=main.py \
    --output-dir="$FINAL_DIR" \
    --output-filename="$MANAGER_EXEC_NAME" \
    --assume-yes-for-downloads --remove-output

# 5. Create the modified multiprocess.json
echo "[5/8] Creating modified process configuration..."
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

# 6. Copy necessary config files
echo "[6/8] Copying configuration files..."
cp ports.toml "$FINAL_DIR/"
if [ -f "models/cfg.json" ]; then
    mkdir -p "$FINAL_DIR/models"
    cp models/cfg.json "$FINAL_DIR/models/"
fi

# 7. Manually copy TTS models directory
echo "[7/8] Copying TTS models..."
mkdir -p "$FINAL_DIR/web/ai_module"
cp -r web/ai_module/tts_models "$FINAL_DIR/web/ai_module/" || { echo "Failed to copy tts_models directory!"; exit 1; }

# 8. Create empty directories that the app expects to exist at runtime
echo "[8/8] Creating runtime directories..."
mkdir -p "$FINAL_DIR/models"
mkdir -p "$FINAL_DIR/temp_uploads"
mkdir -p "$FINAL_DIR/logs"
mkdir -p "$FINAL_DIR/vectordb/rawcontents"
mkdir -p "$FINAL_DIR/web/temp_user_image"

echo "--- Build Finished Successfully! ---"
echo "The complete application is in the '$FINAL_DIR' directory."
echo "To run, navigate into the directory and run the start_app executable:"
echo "cd $FINAL_DIR && ./$MANAGER_EXEC_NAME"