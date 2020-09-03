#!/bin/sh
# Omid55

python3 -m unittest discover -s src/ -p '*_test.py'
# python3 -m unittest discover -s src/ -p '*_test.py' 2>&1 | tee test_result.txt
