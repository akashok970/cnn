#!/bin/bash

# Set data directory
DATADIR="../data"

# Define MNIST file paths
TRAIN_IMAGES="$DATADIR/train-images-idx3-ubyte"
TRAIN_LABELS="$DATADIR/train-labels-idx1-ubyte"
TEST_IMAGES="$DATADIR/t10k-images-idx3-ubyte"
TEST_LABELS="$DATADIR/t10k-labels-idx1-ubyte"

# Path to the binary
MNIST_BIN="./edgeDev"

# Check if binary exists
if [ ! -x "$MNIST_BIN" ]; then
    echo "Error: Binary $MNIST_BIN not found or not executable."
    exit 1
fi

# Check if all MNIST files exist
for file in "$TRAIN_IMAGES" "$TRAIN_LABELS" "$TEST_IMAGES" "$TEST_LABELS"; do
    if [ ! -f "$file" ]; then
        echo "Error: File $file not found."
        exit 1
    fi
done

# Run the MNIST test binary
"$MNIST_BIN" "$TRAIN_IMAGES" "$TRAIN_LABELS" "$TEST_IMAGES" "$TEST_LABELS"

