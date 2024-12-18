#!/bin/bash

mkdir -p font-images-compressed
QUALITY=10    # Default setting - good balance of quality and size

find font-images -type f -name "*.png" -print0 | while IFS= read -r -d '' file; do
    # Get the relative path
    rel_dir=$(dirname "${file#font-images/}")
    
    # Create destination directory
    dest_dir="font-images-compressed/$rel_dir"
    mkdir -p "$dest_dir"
    
    # Generate destination filename
    dest_file="font-images-compressed/$rel_dir/$(basename "${file%.png}").jpg"
    
    # Convert with specified quality
    convert "$file" -quality $QUALITY "$dest_file"
    
    # Optional: Print original and new file sizes for comparison
    orig_size=$(stat -f%z "$file")
    new_size=$(stat -f%z "$dest_file")
    echo "$file: $orig_size bytes -> $dest_file: $new_size bytes"
done