#!/usr/bin/env python3

# Configuration - Change this to match your Google Drive folder ID
drive_folder_id=$CHROMADB_REMOTE_FOLDER_ID
zip_file="chroma_db.zip"
local_folder="chroma_db"

function show_help() {
    echo "Usage: ./loader_vector_db.sh [option]"
    echo "Options:"
    echo "  --save    Compresses and uploads the 'chroma_db' folder to Google Drive"
    echo "  --load    Downloads and extracts the 'chroma_db.zip' file from Google Drive"
    echo "  --help    Displays this help message"
}

function save_to_drive() {
    echo "Compressing the $local_folder folder..."
    zip -r $zip_file $local_folder > /dev/null
    echo "Compression complete."

    echo "Uploading $zip_file to Google Drive..."
    gdrive files upload --parent $drive_folder_id $zip_file
    echo "Upload complete."

    echo "Cleaning up local zip file..."
    rm $zip_file
}

function load_from_drive() {
    echo "Fetching the latest $zip_file from Google Drive..."
    file_id=$(gdrive files list --query "'$drive_folder_id' in parents and name='$zip_file'" --skip-header | awk '{print $1}')
    
    if [[ -z "$file_id" ]]; then
        echo "Error: No file named $zip_file found in the specified Google Drive folder."
        exit 1
    fi

    echo "Downloading $zip_file..."
    gdrive files download $file_id
    echo "Download complete."
    
    echo "Extracting $zip_file..."
    unzip -o $zip_file > /dev/null
    echo "Extraction complete."

    echo "Cleaning up downloaded zip file..."
    rm $zip_file
}

# Check dependencies
dependency_check() {
    if ! command -v gdrive &> /dev/null; then
        echo "Error: 'gdrive' is not installed. Please install it from https://github.com/prasmussen/gdrive."
        exit 1
    fi
    
    echo "Checking Google Drive authentication..."
    echo "Checking Google Drive authentication..."
    if gdrive account current 2>&1 | grep -q "No accounts found"; then
        echo "No Google Drive account linked. Please log in:"
        gdrive account add
    fi
}

dependency_check

# Handle user options
case "$1" in
    --save)
        save_to_drive
        ;;
    --load)
        load_from_drive
        ;;
    --help)
        show_help
        ;;
    *)
        echo "Invalid option. Use --help for usage instructions."
        exit 1
        ;;
esac
