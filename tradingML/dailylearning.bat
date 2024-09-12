
@echo off
python C:\Users\kjets\cs_projects\python-stuff\tradingML\dailylearning.py
IF %ERRORLEVEL% EQU 0 (
    echo Completed successfully
) ELSE (
    echo There was an error
)
pause
