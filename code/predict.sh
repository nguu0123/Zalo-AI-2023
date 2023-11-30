#!/bin/bash
folder1="../results/submission1"
folder2="../results/submission2"
if [ ! -d "$folder1" ]; then
	mkdir -p "$folder1"
fi

if [ ! -d "$folder2" ]; then
	mkdir -p "$folder2"
fi
python3 preprocessing.py
python3 predict.py
