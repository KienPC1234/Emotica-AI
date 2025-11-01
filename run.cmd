@echo off
:: Kịch bản khởi động cho Emotica AI trên Windows

:: --- Các hàm ---

:check_python
    echo [1/4] Dang kiem tra phien ban Python...
    python -c "import sys; exit(0) if (3, 10) <= sys.version_info < (3, 13) else sys.exit(1)"
    if %errorlevel% neq 0 (
        echo Loi: Yeu cau Python 3.10, 3.11 hoac 3.12.
        pause
        exit /b 1
    )
    echo Phien ban Python hop le.
    goto :install_deps

:install_deps
    echo [2/4] Dang kiem tra va cai dat cac thu vien...
    pip install -r requirements.txt
    echo Cac thu vien da duoc cai dat.
    goto :start_servers

:start_servers
    echo [3/4] Khoi dong Llama Server va Web Server...
    start "Llama Server" cmd /c "python llama_server.py"
    start "Web Server" cmd /c "python web/main.py"
    goto :open_browser

:open_browser
    echo [4/4] Doi server khoi dong de mo trinh duyet...
    timeout /t 5 /nobreak > nul

    set "PORT=8000" & rem Mac dinh
    for /f "tokens=1,2,3 delims== " %%i in ('findstr /i "port" ports.toml') do (
        if /i "%%j"=="port" set "PORT=%%k"
    )

    set URL=http://127.0.0.1:%PORT%/chat
    echo Dang mo ung dung tai: %URL%
    start %URL%
    goto :eof


:: --- Kịch bản chính ---
echo --- Bat dau khoi dong Emotica AI ---

call :check_python

echo.
echo --- Emotica AI da san sang ---
echo Cac server dang chay trong cac cua so console rieng biet.
echo De tat ung dung, hay dong cac cua so do.
echo.
pause
