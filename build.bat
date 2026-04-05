@echo off
setlocal

echo ====================================
echo MemorySpine Build — Windows (MSVC)
echo ====================================

where cl >nul 2>nul
if ERRORLEVEL 1 (
    echo MSVC compiler not found. Attempting to initialize Developer Command Prompt...
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
    ) else (
        echo ERROR: Visual Studio build environment not found. Please run this script from a Developer Command Prompt.
        goto :error
    )
)

echo Compiling memspine.dll (for Python bindings) ...
cl /nologo /O2 /EHsc /std:c++17 /LD /Fe:memspine.dll memspine.cpp wininet.lib
if ERRORLEVEL 1 (
    echo.
    echo DLL BUILD FAILED
    goto :error
)

echo.
echo BUILD SUCCESSFUL: memspine.dll
echo.
echo Quick start:
echo   1. pip install -r requirements.txt
echo   2. python app.py
echo.
exit /b 0

:error
exit /b 1
