@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" || exit /b 1

python -c "import genrst; genrst.writeRst()"

REM Notebook conversion step removed from the flattened doc layout.
if exist ".\\_build\\html" rmdir /s /q ".\\_build\\html"

python -m sphinx -b html .\\ .\\_build\\html
set "EXIT_CODE=%ERRORLEVEL%"

popd
exit /b %EXIT_CODE%
