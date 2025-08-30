#!/bin/bash

# Print a hierarchical directory tree with limits
# Usage: ./tree_limited.sh [path]
# Default path is current directory

print_tree() {
    local dir="$1"
    local prefix="$2"
    local depth="$3"

    # List subdirectories (limited to 5)
    local subdirs=($(find "$dir" -maxdepth 1 -mindepth 1 -type d | sort | head -n 5))
    # List files (limited to 4)
    local files=($(find "$dir" -maxdepth 1 -mindepth 1 -type f | sort | head -n 4))

    # Print files first
    for f in "${files[@]}"; do
        echo "${prefix}├── $(basename "$f")"
    done

    # Print subdirectories recursively
    for i in "${!subdirs[@]}"; do
        local sub="${subdirs[$i]}"
        if [ $i -eq $((${#subdirs[@]} - 1)) ]; then
            echo "${prefix}└── $(basename "$sub")"
            print_tree "$sub" "$prefix    " $((depth + 1))
        else
            echo "${prefix}├── $(basename "$sub")"
            print_tree "$sub" "$prefix│   " $((depth + 1))
        fi
    done
}

# Start point
root="${1:-.}"
echo "$(basename "$root")"
print_tree "$root" "" 0

