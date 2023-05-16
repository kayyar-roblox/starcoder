#!/bin/bash
set -e

python -m unittest discover --pattern "*_test.py" --failfast -v
apt-get update
