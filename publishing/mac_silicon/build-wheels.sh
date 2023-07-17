#!/bin/bash
set -e -u -x

cd ../../
maturin build -i python3.10 --release --sdist --strip --out dist
