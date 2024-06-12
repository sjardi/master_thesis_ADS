@ECHO OFF

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


asreview makita template multimodel --classifiers nb logistic --feature_extractors tfidfrelu tfidf tfidfsigmoid tfidfminmax tfidfaddabsmin tfidfcdf
:: asreview makita template multimodel --classifiers nb --feature_extractors tfidf tfidfminmax tfidfzscore tfidflognorm  --query_strategies max
:: asreview makita template multimodel --classifiers logistic nb --feature_extractor tfidfrelu --impossible_models nb,doc2vec logistic,tfidf

call jobs.bat

pause