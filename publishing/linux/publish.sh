#!/bin/bash
set -e -u -x

cd trattoria-core/dist

/opt/python/cp39-cp39/bin/maturin upload --skip-existing *
