#!/bin/bash

# This script is only for removing the first line of every .features file
# Needed because computeIDTFs writes a info message in first line.


for FILE in ../data/out/*.features; do
    for ((i=0; i<=3; i++)); do
        tail -n +2 "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
    done
done
