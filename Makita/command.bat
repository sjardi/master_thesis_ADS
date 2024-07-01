@ECHO OFF

IF NOT EXIST "sbert_all\command_main.bat" (
    ECHO First batch file not found!
    PAUSE
    EXIT /B
)

IF NOT EXIST "%~dp0doc2vec_all\command_main.bat" (
    ECHO First batch file not found!
    PAUSE
    EXIT /B
)

REM Run the first batch file
ECHO Running the first batch file...
CALL "%~dp0doc2vec_all\command_main.bat"

REM Check if the second batch file exists
IF NOT EXIST "%~dp0tfidf_all\command_main.bat" (
    ECHO Second batch file not found!
    PAUSE
    EXIT /B
)

REM Run the second batch file
ECHO Running the second batch file...
CALL "%~dp0tfidf_all\command_main.bat"

PAUSE
