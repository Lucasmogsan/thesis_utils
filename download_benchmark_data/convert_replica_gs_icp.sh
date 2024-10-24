#!/bin/bash

# Loop through each office directory
for OFFICE_DIR in dataset/Replica/office{0..4}; do
    SOURCE_DIR="$OFFICE_DIR/results"
    DEPTH_DIR="$OFFICE_DIR/depth_images"
    RGB_DIR="$OFFICE_DIR/images"

    # Create target directories if they don't exist
    mkdir -p "$DEPTH_DIR"
    mkdir -p "$RGB_DIR"

    # Move depth images if they exist
    if ls "$SOURCE_DIR"/depth*.png 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR"/depth*.png "$DEPTH_DIR/"
        echo "Moved depth images for $OFFICE_DIR!"
    else
        echo "No depth images found in $SOURCE_DIR for $OFFICE_DIR."
    fi

    # Move RGB images if they exist
    if ls "$SOURCE_DIR"/frame*.jpg 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR"/frame*.jpg "$RGB_DIR/"
        echo "Moved RGB images for $OFFICE_DIR!"
    else
        echo "No RGB images found in $SOURCE_DIR for $OFFICE_DIR."
    fi
done

# Loop through each room directory
for ROOM_DIR in dataset/Replica/room{0..2}; do
    SOURCE_DIR="$ROOM_DIR/results"
    DEPTH_DIR="$ROOM_DIR/depth_images"
    RGB_DIR="$ROOM_DIR/images"

    # Create target directories if they don't exist
    mkdir -p "$DEPTH_DIR"
    mkdir -p "$RGB_DIR"

    # Move depth images if they exist
    if ls "$SOURCE_DIR"/depth*.png 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR"/depth*.png "$DEPTH_DIR/"
        echo "Moved depth images for $ROOM_DIR!"
    else
        echo "No depth images found in $SOURCE_DIR for $ROOM_DIR."
    fi

    # Move RGB images if they exist
    if ls "$SOURCE_DIR"/frame*.jpg 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR"/frame*.jpg "$RGB_DIR/"
        echo "Moved RGB images for $ROOM_DIR!"
    else
        echo "No RGB images found in $SOURCE_DIR for $ROOM_DIR."
    fi
done
