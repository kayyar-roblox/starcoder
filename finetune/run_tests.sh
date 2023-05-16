#!/bin/bash
if [ -d finetune ]; then
    cd finetune
fi

python -m unittest discover --pattern "*_test.py" --failfast -v
