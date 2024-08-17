@echo off
python C:\Users\kjets\cs_projects\python-stuff\options_ML\yesterdayoptiondata.py
IF %ERRORLEVEL% EQU 0 (
    echo Completed successfully
) ELSE (
    echo There was an error
)
pause
