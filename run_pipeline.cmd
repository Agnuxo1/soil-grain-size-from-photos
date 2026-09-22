@echo off
setlocal
if "%1"=="" goto :help
python src\soil_gsd_pipeline.py %*
goto :eof
:help
echo Usage:
echo   run_pipeline.cmd inspect --data-dir data
echo   run_pipeline.cmd train --data-dir data --output-dir artifacts\soil_gsd
echo   run_pipeline.cmd predict --data-dir data --model artifacts\soil_gsd\model.joblib --output artifacts\soil_gsd\submission.csv
echo   run_pipeline.cmd smoke-test
