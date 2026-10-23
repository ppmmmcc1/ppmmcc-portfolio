@echo off
rem nqlab overnight batch. Schedule with Windows Task Scheduler (see README).
rem Drains the job queue: queued families through the funnel (never the holdout),
rem baseline reports, then the morning summary.
cd /d %~dp0\..
call .venv\Scripts\activate.bat
rem UTF-8 for every file and console stream, whatever the Windows code page.
set PYTHONUTF8=1
rem Locale-independent date (the %DATE% slicing in the spec breaks outside en-US).
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set NQDATE=%%i
if not exist reports mkdir reports
rem run-queue itself asks Windows not to sleep while it runs (SetThreadExecutionState).
python -m nqlab.cli run-queue --workers auto --log reports\overnight_%NQDATE%.log
exit /b %ERRORLEVEL%
