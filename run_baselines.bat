@echo off
setlocal

set OUTFILE=baseline_results.txt

echo Running all 4 inference baselines... > %OUTFILE%
echo Started: %date% %time% >> %OUTFILE%
echo ============================================================ >> %OUTFILE%

echo [1/4] R1 Rule-based...
echo. >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo RUN 1: inference.py -- RULE-BASED >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
set USE_LLM=0
set LLAMA_BASELINE=0
python inference.py >> %OUTFILE% 2>&1

echo [2/4] R1 Llama baseline...
echo. >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo RUN 2: inference.py -- LLAMA BASELINE >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
set USE_LLM=1
set LLAMA_BASELINE=1
python inference.py >> %OUTFILE% 2>&1

echo [3/4] R2 Rule-based...
echo. >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo RUN 3: inference_r2.py -- RULE-BASED >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
set USE_LLM=0
set LLAMA_BASELINE=0
python inference_r2.py >> %OUTFILE% 2>&1

echo [4/4] R2 Llama baseline...
echo. >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo RUN 4: inference_r2.py -- LLAMA BASELINE >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
set USE_LLM=1
set LLAMA_BASELINE=1
python inference_r2.py >> %OUTFILE% 2>&1

echo. >> %OUTFILE%
echo ============================================================ >> %OUTFILE%
echo Finished: %date% %time% >> %OUTFILE%

echo.
echo All done! Results saved to %OUTFILE%
endlocal