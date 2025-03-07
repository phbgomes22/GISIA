import os
import gdown
import zipfile


def download_vector_database():

    # Google Drive file ID (Extracted from your link)
    file_id = os.getenv("CHROMADB_REMOTE_FILE_ID")

    # Output directory
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    # File path for the downloaded ZIP file
    zip_file_path = os.path.join(output_dir, "downloaded.zip")

    # Construct the gdown download URL
    file_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the ZIP file
    gdown.download(file_url, zip_file_path, quiet=False)

    # Extract the ZIP file
    if zip_file_path.endswith(".zip"):
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_file_path)  # Remove ZIP file after extraction

    print("✅ Download and extraction of Vector DB complete. Files are in the 'chroma_db' folder.")
