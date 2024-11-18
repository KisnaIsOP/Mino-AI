@echo off
setlocal enabledelayedexpansion

:: Set colors for console output
color 0A

:menu
cls
echo [92m=== Mino AI Management Console ===[0m
echo.
echo [96m1.[0m Start Mino AI with Live Logs
echo [96m2.[0m View Current Logs
echo [96m3.[0m Stop Mino AI
echo [96m4.[0m Hot Reload Mino Components
echo [96m5.[0m Manage Mino Settings
echo [96m6.[0m Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto start_server
if "%choice%"=="2" goto view_logs
if "%choice%"=="3" goto stop_server
if "%choice%"=="4" goto hot_reload
if "%choice%"=="5" goto manage_ai
if "%choice%"=="6" goto end
goto menu

:start_server
cls
:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:: Set log file name with timestamp
set TIMESTAMP=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=!TIMESTAMP: =0!

:: Define log files
set MAIN_LOG=logs\mino_%TIMESTAMP%.log
set ERROR_LOG=logs\errors_%TIMESTAMP%.log
set CHAT_LOG=logs\chat_%TIMESTAMP%.log
set RELOAD_LOG=logs\reload_%TIMESTAMP%.log

:: Save PID for server management
echo [92mStarting Mino AI server...[0m
start /B python app.py > %MAIN_LOG% 2> %ERROR_LOG%
for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe"') do set SERVER_PID=%%a
echo %SERVER_PID% > logs\server.pid

:: Start live log monitoring in separate windows
start "Main Log Monitor" cmd /c "echo [92mMain Log Monitor[0m && powershell Get-Content -Path '%MAIN_LOG%' -Wait -Tail 10"
start "Error Log Monitor" cmd /c "echo [91mError Log Monitor[0m && powershell Get-Content -Path '%ERROR_LOG%' -Wait -Tail 10"
start "Chat Log Monitor" cmd /c "echo [93mChat Log Monitor[0m && powershell Get-Content -Path '%CHAT_LOG%' -Wait -Tail 10"

echo [92mServer started successfully![0m
echo Server PID: %SERVER_PID%
echo.
echo Press any key to return to menu...
pause > nul
goto menu

:view_logs
cls
echo [96m=== Current Logs ===[0m
echo.
if exist "logs\server.pid" (
    set /p SERVER_PID=<logs\server.pid
    echo Server Status: [92mRUNNING[0m (PID: %SERVER_PID%)
) else (
    echo Server Status: [91mSTOPPED[0m
)
echo.
echo [93m1.[0m View Main Log
echo [93m2.[0m View Error Log
echo [93m3.[0m View Chat Log
echo [93m4.[0m Return to Main Menu
echo.
set /p log_choice="Enter your choice (1-4): "

if "%log_choice%"=="1" goto view_main_log
if "%log_choice%"=="2" goto view_error_log
if "%log_choice%"=="3" goto view_chat_log
if "%log_choice%"=="4" goto menu
goto view_logs

:view_main_log
start "Main Log Monitor" cmd /c "echo [92mMain Log Monitor[0m && powershell Get-Content -Path '%MAIN_LOG%' -Wait -Tail 10"
goto view_logs

:view_error_log
start "Error Log Monitor" cmd /c "echo [91mError Log Monitor[0m && powershell Get-Content -Path '%ERROR_LOG%' -Wait -Tail 10"
goto view_logs

:view_chat_log
start "Chat Log Monitor" cmd /c "echo [93mChat Log Monitor[0m && powershell Get-Content -Path '%CHAT_LOG%' -Wait -Tail 10"
goto view_logs

:stop_server
cls
if exist "logs\server.pid" (
    set /p SERVER_PID=<logs\server.pid
    echo [91mStopping server (PID: %SERVER_PID%)...[0m
    taskkill /PID %SERVER_PID% /F
    del logs\server.pid
    echo [92mServer stopped successfully![0m
) else (
    echo [93mNo running server found.[0m
)
echo.
echo Press any key to return to menu...
pause > nul
goto menu

:hot_reload
cls
echo [96m=== Hot Reload Mino Components ===[0m
echo.
if not exist "logs\server.pid" (
    echo [91mServer is not running. Start the server first.[0m
    echo.
    echo Press any key to return to menu...
    pause > nul
    goto menu
)

echo [93mSelect component to reload:[0m
echo 1. Reload AI Model
echo 2. Reload Chat Analysis
echo 3. Reload Emotion Detection
echo 4. Reload All Components
echo 5. Return to Main Menu
echo.
set /p reload_choice="Enter your choice (1-5): "

if "%reload_choice%"=="1" (
    echo [92mReloading AI Model...[0m
    python -c "from simple_analysis import reload_ai_model; reload_ai_model()" > %RELOAD_LOG% 2>&1
) else if "%reload_choice%"=="2" (
    echo [92mReloading Chat Analysis...[0m
    python -c "from simple_analysis import reload_chat_analysis; reload_chat_analysis()" > %RELOAD_LOG% 2>&1
) else if "%reload_choice%"=="3" (
    echo [92mReloading Emotion Detection...[0m
    python -c "from simple_analysis import reload_emotion_detection; reload_emotion_detection()" > %RELOAD_LOG% 2>&1
) else if "%reload_choice%"=="4" (
    echo [92mReloading All Components...[0m
    python -c "from simple_analysis import reload_all; reload_all()" > %RELOAD_LOG% 2>&1
) else if "%reload_choice%"=="5" (
    goto menu
)

echo.
echo [92mReload Complete! Check reload logs for details.[0m
timeout /t 2 > nul
goto hot_reload

:manage_ai
cls
echo [96m=== Manage Mino Settings ===[0m
echo.
if not exist "logs\server.pid" (
    echo [91mMino AI is not running. Start the server first.[0m
    echo.
    echo Press any key to return to menu...
    pause > nul
    goto menu
)

echo [93mSelect setting to modify:[0m
echo 1. Update Response Style
echo 2. Adjust Emotion Sensitivity
echo 3. Modify Context Window
echo 4. Change Mino Version
echo 5. View Current Settings
echo 6. Return to Main Menu
echo.
set /p settings_choice="Enter your choice (1-6): "

if "%settings_choice%"=="1" (
    echo [92mCurrent Response Styles:[0m
    echo 1. Professional
    echo 2. Casual
    echo 3. Technical
    echo 4. Friendly
    set /p style="Select style (1-4): "
    python -c "from simple_analysis import update_response_style; update_response_style('%style%')"
) else if "%settings_choice%"=="2" (
    echo [92mEmotion Sensitivity (1-10):[0m
    set /p sensitivity="Enter sensitivity level: "
    python -c "from simple_analysis import update_emotion_sensitivity; update_emotion_sensitivity(%sensitivity%)"
) else if "%settings_choice%"=="3" (
    echo [92mContext Window Size (1-20):[0m
    set /p window="Enter window size: "
    python -c "from simple_analysis import update_context_window; update_context_window(%window%)"
) else if "%settings_choice%"=="4" (
    echo [92mAvailable Mino Versions:[0m
    echo 1. Mino Standard
    echo 2. Mino Enhanced
    echo 3. Mino Custom
    set /p model="Select version (1-3): "
    python -c "from simple_analysis import change_language_model; change_language_model('%model%')"
) else if "%settings_choice%"=="5" (
    python -c "from simple_analysis import display_current_settings; display_current_settings()"
    echo.
    echo Press any key to continue...
    pause > nul
) else if "%settings_choice%"=="6" (
    goto menu
)

echo.
echo [92mSettings updated successfully![0m
timeout /t 2 > nul
goto manage_ai

:end
echo [92mThank you for using Mino AI![0m
timeout /t 2 > nul
exit
