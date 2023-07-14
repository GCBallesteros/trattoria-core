#!/bin/bash
set -e -u -x

cd ../../dist

maturin upload --skip-existing *
