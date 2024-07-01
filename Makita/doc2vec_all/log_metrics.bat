@ECHO OFF
setlocal enabledelayedexpansion
echo Setting non-interactive backend
set MPLBACKEND=Agg

REM Log file for GPU metrics
set "log_file=system_metrics.log"

REM Interval in seconds
set "interval=1"

REM Clear previous log
IF EXIST %log_file% DEL %log_file%

REM Function to log GPU metrics
:LogMetrics
REM Infinite loop to continuously log GPU metrics
:LogLoop
    REM Get GPU utilization and memory used
    echo Time start:  !TIME! >> %log_file%
    @REM for /F "tokens=*" %%a in ('nvidia-smi --query-gpu=utilization.memory --format=csv') do (
    @REM     echo Raw output: %%a >> %log_file%
    @REM )
    nvidia-smi --query-gpu=utilization.memory --format=csv >> %log_file%
    nvidia-smi --query-gpu=utilization.gpu --format=csv >> %log_file%
    @REM  for /F "tokens=*" %%a in ('nvidia-smi --query-gpu=utilization.gpu --format=csv') do (
    @REM     echo !TIME! Raw output: %%a >> %log_file%
    @REM )

    REM Sleep for the interval
    timeout /t %interval% > nul
    echo Time end:  !TIME! >> %log_file%

goto :LogLoop

endlocal
exit /b