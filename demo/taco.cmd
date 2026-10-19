@echo off
rem Runs the Windows taco program with the model file of this folder.
rem Use it as the command: taco fold notes.txt, taco unfold notes.txt.taco, taco --help.
setlocal
if not exist "%~dp0gpt2.hsm" (
  echo taco demo: the model file gpt2.hsm is not in %~dp0 1>&2
  exit /b 1
)
set "TACO_MODEL=%~dp0gpt2.hsm"
"%~dp0bin\windows-x86_64\taco.exe" %*
exit /b %ERRORLEVEL%
