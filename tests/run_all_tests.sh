#!/bin/bash

DIR=../build/tests/

if [ -z "$DIR" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

echo "Scanning directory: $DIR"
echo

# 先检索 ELF 可执行文件
binaries=$(find "$DIR" -type f -executable | while read file; do
    if file "$file" | grep -q "ELF"; then
        echo "$file"
    fi
done)

if [ -z "$binaries" ]; then
    echo "No ELF binaries found."
    exit 0
fi

echo "Found binaries:"
echo "$binaries"
echo "=============================="
echo

# 逐个执行
for bin in $binaries; do
    echo "===== Running: $bin ====="

    "$bin"
    ret=$?

    echo "Return code: $ret"

    if [ $ret -eq 0 ]; then
        echo "Result: PASS"
    else
        echo "Result: FAIL"
    fi

    echo
done

echo "All done."
