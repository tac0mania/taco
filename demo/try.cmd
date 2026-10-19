@echo off
rem One command: folds and unfolds the sample file, checks the result, and prints the hash of
rem the .taco file. That hash is the same on every machine and every backend.
setlocal
set "HERE=%~dp0"
set "WORK=%TEMP%\taco-demo-%RANDOM%"
mkdir "%WORK%"
echo Taco demo: fold and unfold the sample file. This takes about 15 s to 1 minute.
echo.
echo Fold started at %TIME%
rem The profile text runs fold on the GPU too, so the demo tests the GPU in both directions.
call "%HERE%taco.cmd" fold "%HERE%samples\sample.txt" -o "%WORK%\sample.txt.taco" -f --profile text %*
if errorlevel 1 goto failed
echo Unfold started at %TIME%
call "%HERE%taco.cmd" unfold "%WORK%\sample.txt.taco" -o "%WORK%\sample.txt" -f %*
if errorlevel 1 goto failed
echo Unfold ended at %TIME%
echo.
fc /b "%HERE%samples\sample.txt" "%WORK%\sample.txt" >nul
if errorlevel 1 goto failed
echo PASS: the file came back byte for byte.
echo SHA-256 of the .taco file:
certutil -hashfile "%WORK%\sample.txt.taco" SHA256 | findstr /v ":"
echo Compare it with the value in README.md. It must be the same on every machine.
rmdir /s /q "%WORK%"
exit /b 0
:failed
echo FAIL: see the message above. Please report it.
rmdir /s /q "%WORK%"
exit /b 1
