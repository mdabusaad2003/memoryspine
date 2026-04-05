#!/usr/bin/env bash

echo "===================================="
echo "MemorySpine Build — Linux/macOS"
echo "===================================="

CXX="${CXX:-g++}"

if ! command -v "$CXX" &> /dev/null; then
    CXX="clang++"
fi

if ! command -v "$CXX" &> /dev/null; then
    echo "ERROR: Neither g++ nor clang++ found. Please install a C++ compiler."
    exit 1
fi

echo "Compiling libmemspine.so (for Python bindings) with $CXX..."

# Mac uses .dylib normally, but Python ctypes can load .so or .dylib. We'll stick to .so
$CXX -O2 -fPIC -shared -std=c++17 -o libmemspine.so memspine.cpp

if [ $? -ne 0 ]; then
    echo "DLL BUILD FAILED"
    exit 1
fi

echo ""
echo "BUILD SUCCESSFUL: libmemspine.so"
echo ""
echo "Quick start:"
echo "  1. pip install -r requirements.txt"
echo "  2. python app.py"
echo ""
