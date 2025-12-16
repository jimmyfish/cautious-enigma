#!/bin/bash

# Check if symbol argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {symbol}"
    echo "Example: $0 ADMR"
    exit 1
fi

SYMBOL="$1"
SOURCES_DIR="sources"
SYMBOL_DIR="${SOURCES_DIR}/${SYMBOL}"

# Create sources directory if it doesn't exist
mkdir -p "${SOURCES_DIR}"

# Create symbol directory if it doesn't exist
mkdir -p "${SYMBOL_DIR}"

# Find the highest number in the symbol directory
# If no numbered directories exist, start with 1
MAX_NUM=0
if [ -d "${SYMBOL_DIR}" ]; then
    for dir in "${SYMBOL_DIR}"/*; do
        if [ -d "${dir}" ]; then
            dirname=$(basename "${dir}")
            # Check if the directory name is a number
            if [[ "${dirname}" =~ ^[0-9]+$ ]]; then
                if [ "${dirname}" -gt "${MAX_NUM}" ]; then
                    MAX_NUM="${dirname}"
                fi
            fi
        fi
    done
fi

# Increment the number
NEW_NUM=$((MAX_NUM + 1))

# Create the new directory
NEW_DIR="${SYMBOL_DIR}/${NEW_NUM}"
mkdir -p "${NEW_DIR}"

# Create empty JSON files
touch "${NEW_DIR}/market-detector.json"
touch "${NEW_DIR}/price-feed.json"
touch "${NEW_DIR}/running-trade.json"
touch "${NEW_DIR}/today-running-trade.json"
touch "${NEW_DIR}/orderbook.json"

echo "Created directory: ${NEW_DIR}"
echo "Created JSON files:"
echo "  - market-detector.json"
echo "  - price-feed.json"
echo "  - running-trade.json"
echo "  - today-running-trade.json"
echo "  - orderbook.json"

