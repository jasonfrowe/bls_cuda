#!/bin/sh
# Build libcbls.so next to this script.
cd "$(dirname "$0")" && gcc -O3 -march=x86-64-v2 -fopenmp -shared -fPIC -Wall cbls.c -o libcbls.so -lm
