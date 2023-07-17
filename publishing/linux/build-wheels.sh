#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

git clone https://github.com/GCBallesteros/trattoria-core
cd trattoria-core

/opt/python/cp39-cp39/bin/maturin build -i python3.8 --release --sdist --strip --out dist
/opt/python/cp39-cp39/bin/maturin build -i python3.9 --release --sdist --strip --out dist
/opt/python/cp39-cp39/bin/maturin build -i python3.10 --release --sdist --strip --out dist
/opt/python/cp39-cp39/bin/maturin build -i python3.11 --release --sdist --strip --out dist
