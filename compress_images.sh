#!/bin/bash

# Create base output directory
# mkdir -p compressed-images

# # Convert and compress JPG/JPEG files, preserving directory structure
# find font-images -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -exec sh -c '
#     # Get the relative path of the file
#     rel_path=${1#font-images/}
#     # Create the target directory
#     mkdir -p "compressed-images/$(dirname "$rel_path")"
#     # Compress the image
#     sips -s format jpeg -s formatOptions 30 "$1" --out "compressed-images/$rel_path"
# ' sh {} \;

# # Convert and compress PNG files, preserving directory structure
# find font-images -type f -iname "*.png" -exec sh -c '
#     # Get the relative path of the file
#     rel_path=${1#font-images/}
#     # Create the target directory
#     mkdir -p "compressed-images/$(dirname "$rel_path")"
#     # Compress the image
#     sips -s format png -s formatOptions 5 "$1" --out "compressed-images/$rel_path"
# ' sh {} \;

# # Create compressed zip file
# zip -9 -r compressed-font-images.zip compressed-images/



#!/bin/bash
# Create base output directory
# mkdir -p compressed-images

# # Convert and compress JPG/JPEG/PNG files
# find font-images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec sh -c '
#     # Get the relative path of the file
#     rel_path=${1#font-images/}
#     # Create the target directory
#     mkdir -p "compressed-images/$(dirname "$rel_path")"
#     # Compress the image with aggressive settings
#     convert "$1" -quality 5 -resize 50% "compressed-images/$rel_path"
# ' sh {} \;

# # Create compressed zip file
# zip -9 -r compressed-font-images.zip compressed-images/

#!/bin/bash

# First, install ImageMagick if not already installed
# brew install imagemagick

# Create base output directory
mkdir -p compressed-images

# Convert and compress JPG/JPEG/PNG files
find font-images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec sh -c '
    # Get the relative path of the file
    rel_path=${1#font-images/}
    # Create the target directory
    mkdir -p "compressed-images/$(dirname "$rel_path")"
    
    # Determine if file is PNG
    if [[ "$1" == *.png ]]; then
        # For PNGs, preserve colorspace and use PNG-specific compression
        magick "$1" -strip -resize 50% -quality 20 -define png:compression-level=9 "compressed-images/$rel_path"
    else
        # For JPEGs, use regular compression
        magick "$1" -strip -resize 50% -quality 20 "compressed-images/$rel_path"
    fi
' sh {} \;

# Create compressed zip file
zip -9 -r compressed-font-images.zip compressed-images/