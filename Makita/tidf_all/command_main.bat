@ECHO OFF
setlocal enabledelayedexpansion

REM Set the matplotlib backend to Agg
@REM set MPLBACKEND=Agg

REM Output files
set "output_file=results.log"
set "error_file=errors.log"

REM Clear previous logs
IF EXIST %output_file% DEL %output_file%
IF EXIST %error_file% DEL %error_file%
IF EXIST system_metrics.log DEL system_metrics.log

REM Start the logging script
start "Logging" /B cmd /c "log_metrics.bat" 

REM Define the classifiers and feature extractors
set "classifiers=logistic nb"
@REM set "feature_extractors= tfidf_pareto_minmax tfidf_l2_normalize_sigmoid  tfidfsigmoid tfidfminmax tfidf_absmin tfidfcdf tfidf_zscore_minmax tfidf_l2_normalize_minmax tfidf_zscore_absmin tfidf_pareto_absmin tfidf_l2_normalize_absmin tfidf_zscore_cdf tfidf_pareto_cdf tfidf_l2_normalize_cdf tfidf_zscore_sigmoid tfidf_pareto_sigmoid tfidfn tfidf_l2_normalize"
set "feature_extractors= tfidfsigmoid tfidfcdf tfidf_zscore_absmin tfidf_pareto_absmin tfidf_zscore_cdf tfidf_pareto_cdf tfidf_l2_normalize_cdf tfidf_zscore_sigmoid tfidf_pareto_sigmoid "

REM Loop through each classifier and feature extractor
for %%c in (%classifiers%) do (
    for %%f in (%feature_extractors%) do (
         
        REM Capture start time
        for /F "tokens=1-4 delims=:.," %%a in ("!TIME!") do (
            set "START_HOUR=%%a"
            set "START_MINUTE=%%b"
            set "START_SECOND=%%c"
            set "START_CSEC=%%d"
        )

        echo Running classifier: %%c with feature extractor: %%f
        echo Running classifier: %%c with feature extractor: %%f >> %output_file%
        echo Start Time: !START_HOUR!:!START_MINUTE!:!START_SECOND!,!START_CSEC! >> %output_file%
        
        REM Cleanup
        REM Check for and remove the "output" folder
        if exist "output" (
            echo Removing "output" folder...
            rmdir /s /q "output" 
            
        ) else (
            echo "output" folder does not exist.
        )

        REM Check for and remove the "scripts" folder
        if exist "scripts" (
            echo Removing "scripts" folder...
            rmdir /s /q "scripts" 
            
        ) else (
            echo "scripts" folder does not exist.
        )

        REM Check for and remove the "jobs" batch file
        if exist "jobs.bat" (
            echo Removing "jobs.bat" file...
            del /q "jobs.bat" 
            
        ) else (
            echo "jobs.bat" file does not exist.
        )

        REM Check for and remove the "README" markdown file
        if exist "README.md" (
            echo Removing "README.md" file...
            del /q "README.md" 
            
        ) else (
            echo "README.md" file does not exist.
        )

        echo Cleanup complete.

        REM Run the asreview command
        echo Running asreview command...
        asreview makita template multimodel --classifiers %%c --feature_extractors %%f 

        call jobs.bat

        REM Capture end time
        for /F "tokens=1-4 delims=:.," %%a in ("!TIME!") do (
            set "END_HOUR=%%a"
            set "END_MINUTE=%%b"
            set "END_SECOND=%%c"
            set "END_CSEC=%%d"
        )

        REM Debugging: Display captured times
        echo START_HOUR: !START_HOUR!, START_MINUTE: !START_MINUTE!, START_SECOND: !START_SECOND!, START_CSEC!
        echo END_HOUR: !END_HOUR!, END_MINUTE: !END_MINUTE!, END_SECOND: !END_SECOND!, END_CSEC!

        REM Convert start and end times to total centiseconds
        set /A "START_TOTAL_CSEC=START_HOUR*360000 + START_MINUTE*6000 + START_SECOND*100 + START_CSEC"
        set /A "END_TOTAL_CSEC=END_HOUR*360000 + END_MINUTE*6000 + END_SECOND*100 + END_CSEC"

        REM Debugging: Display total centiseconds
        echo START_TOTAL_CSEC: !START_TOTAL_CSEC!
        echo END_TOTAL_CSEC: !END_TOTAL_CSEC!

        REM Calculate the difference in centiseconds
        set /A "DIFF_CSEC=END_TOTAL_CSEC-START_TOTAL_CSEC"
        if !DIFF_CSEC! LSS 0 set /A "DIFF_CSEC+=8640000"

        REM Convert centiseconds to seconds for output
        set /A "DIFF_SECONDS=DIFF_CSEC / 100"
        set /A "DIFF_MINUTES=DIFF_SECONDS / 60"

        REM Output the difference
        echo Runtime for classifier %%c with feature extractor %%f: !DIFF_SECONDS! seconds >> %output_file%
        echo Start Total Centiseconds: !START_TOTAL_CSEC!, End Total Centiseconds: !END_TOTAL_CSEC!, Difference in Seconds: !DIFF_SECONDS! >> %output_file%
        echo End Time: !END_HOUR!:!END_MINUTE!:!END_SECOND!,!END_CSEC! >> %output_file%
        echo ############## The whole computation took !DIFF_SECONDS! SECONDS or !DIFF_MINUTES! MINUTES ###################### >> %output_file%
    )
)

REM Stop the logging script
taskkill /FI "WINDOWTITLE eq Logging" /F > nul 

@REM Shut down the system
shutdown /s /f /t 0

@REM pause

echo DONE >> %output_file%
endlocal
