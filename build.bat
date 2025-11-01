@echo off
:: --- Windows Build Script for Multi-Process Application (v5) ---
:: Changes:
:: - Removed all --enable-plugin flags as requested.
:: - Tries to build with static libpython first, falls back to non-static if it fails.
:: - Downloads two specified GGUF models into the build directory.
:: - Manually copies the tts_models directory.

echo --- Starting Advanced Windows Build (v5) ---

:: --- Configuration ---
set FINAL_DIR=build_output
set LLAMA_EXEC_NAME=llama_server_exec
set WEB_EXEC_NAME=web_server_exec
set MANAGER_EXEC_NAME=start_app

set URL_LLAMA2=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true
set URL_VISTRAL=https://huggingface.co/janhq/Vistral-7b-Chat-GGUF/resolve/main/vitral-7b-chat.Q4_K_M.gguf?download=true

:: --- Build Steps ---

echo [1/9] Cleaning up previous build directory...
if exist %FINAL_DIR% ( rd /s /q %FINAL_DIR% )
mkdir %FINAL_DIR%

:: --- Data File Exclusion ---
:: Note: tts_models is now copied manually, so it's removed from this list.
set NUITKA_DATA_EXCLUDES=--noinclude-data-files=models/** --noinclude-data-files=vectordb/** --noinclude-data-files=embedding_models/** --noinclude-data-files=image_captioners/** --noinclude-data-files=logs/** --noinclude-data-files=temp_uploads/**

:: --- Compilation Section ---

:: 2. Compile llama_server.py
echo [2/9] Compiling Llama Server...
echo --^> Attempting to compile with static libpython...
python -m nuitka --main=llama_server.py --static-libpython=yes --output-dir=%FINAL_DIR% --output-filename=%LLAMA_EXEC_NAME% %NUITKA_DATA_EXCLUDES% --assume-yes-for-downloads --remove-output
if %errorlevel% neq 0 (
    echo --^> Static build failed. Retrying without static libpython...
    python -m nuitka --main=llama_server.py --static-libpython=no --output-dir=%FINAL_DIR% --output-filename=%LLAMA_EXEC_NAME% %NUITKA_DATA_EXCLUDES% --assume-yes-for-downloads --remove-output
)
if %errorlevel% neq 0 ( echo FATAL: Llama Server compilation failed! & exit /b 1 )

:: 3. Compile web/main.py
echo [3/9] Compiling Web Server...
echo --^> Attempting to compile with static libpython...
python -m nuitka --main=web/main.py --static-libpython=yes --output-dir=%FINAL_DIR% --output-filename=%WEB_EXEC_NAME% %NUITKA_DATA_EXCLUDES% --include-data-dir=web/templates=web/templates --include-data-dir=web/statics=web/statics --assume-yes-for-downloads --remove-output
if %errorlevel% neq 0 (
    echo --^> Static build failed. Retrying without static libpython...
    python -m nuitka --main=web/main.py --static-libpython=no --output-dir=%FINAL_DIR% --output-filename=%WEB_EXEC_NAME% %NUITKA_DATA_EXCLUDES% --include-data-dir=web/templates=web/templates --include-data-dir=web/statics=web/statics --assume-yes-for-downloads --remove-output
)
if %errorlevel% neq 0 ( echo FATAL: Web Server compilation failed! & exit /b 1 )

:: 4. Compile main.py (Process Manager)
echo [4/9] Compiling Process Manager...
echo --^> Attempting to compile with static libpython...
python -m nuitka --main=main.py --static-libpython=yes --output-dir=%FINAL_DIR% --output-filename=%MANAGER_EXEC_NAME% --assume-yes-for-downloads --remove-output
if %errorlevel% neq 0 (
    echo --^> Static build failed. Retrying without static libpython...
    python -m nuitka --main=main.py --static-libpython=no --output-dir=%FINAL_DIR% --output-filename=%MANAGER_EXEC_NAME% --assume-yes-for-downloads --remove-output
)
if %errorlevel% neq 0 ( echo FATAL: Process Manager compilation failed! & exit /b 1 )

:: 5. Create the modified multiprocess.json
echo [5/9] Creating modified process configuration...
(
  echo {
  echo   "llama_server": [
  echo     ".\\%LLAMA_EXEC_NAME%.exe"
  echo   ],
  echo   "web_server": [
  echo     ".\\%WEB_EXEC_NAME%.exe"
  echo   ]
  echo }
) > "%FINAL_DIR%\multiprocess.json"

:: 6. Download GGUF Models
echo [6/9] Downloading GGUF models (this may take a while)...
mkdir "%FINAL_DIR%\models"
echo --^> Downloading Llama-2-7B-Chat...
curl -L "%URL_LLAMA2%" -o "%FINAL_DIR%\models\llama-2-7b-chat.Q4_K_M.gguf"
if %errorlevel% neq 0 ( echo FATAL: Failed to download Llama-2 model! & exit /b 1 )
echo --^> Downloading Vistral-7B-Chat...
curl -L "%URL_VISTRAL%" -o "%FINAL_DIR%\models\vitral-7b-chat.Q4_K_M.gguf"
if %errorlevel% neq 0 ( echo FATAL: Failed to download Vistral model! & exit /b 1 )

:: 7. Copy necessary config files
echo [7/9] Copying configuration files...
copy ports.toml "%FINAL_DIR%\"
if exist "models\cfg.json" (
    copy "models\cfg.json" "%FINAL_DIR%\models\"
)

:: 8. Manually copy TTS models directory
echo [8/9] Copying TTS models...
xcopy /E /I /Y web\ai_module\tts_models "%FINAL_DIR%\web\ai_module\tts_models\"
if %errorlevel% neq 0 ( echo FATAL: Failed to copy tts_models directory! & exit /b 1 )

:: 9. Create empty directories that the app expects to exist at runtime
echo [9/9] Creating runtime directories...
mkdir "%FINAL_DIR%\temp_uploads"
mkdir "%FINAL_DIR%\logs"
mkdir "%FINAL_DIR%\vectordb\rawcontents"
mkdir "%FINAL_DIR%\web\temp_user_image"

echo --- Build Finished Successfully! ---
echo The complete application is in the '%FINAL_DIR%' directory.
echo To run, navigate into the directory and run the start_app executable:
echo cd %FINAL_DIR% ^& %MANAGER_EXEC_NAME%.exe