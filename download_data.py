import os
import requests
import zipfile
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    filename='dataset_setup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
DOWNLOAD_CONFIG = {
    'downloads_folder': './downloaded/',
    'datasets_folder': './data/',
    'download_urls': {
        'ak_captions': 'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip',
        'flickr8k': 'https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k',
        'flickr30k': 'https://www.kaggle.com/api/v1/datasets/download/hsankesara/flickr-image-dataset',
        'coco_train2014': 'http://images.cocodataset.org/zips/train2014.zip',
        'coco_test2014': 'http://images.cocodataset.org/zips/test2014.zip',
        'coco_val2014': 'http://images.cocodataset.org/zips/val2014.zip',
        'glove': 'https://nlp.stanford.edu/data/glove.6B.zip',  # Added GloVe
    },
    'download_destinations': {
        'ak_captions': 'caption_datasets.zip',
        'flickr8k': 'flickr8k_dataset.zip',
        'flickr30k': 'flickr30k_dataset.zip',
        'coco_train2014': 'coco_train2014.zip',
        'coco_test2014': 'coco_test2014.zip',
        'coco_val2014': 'coco_val2014.zip',
        'glove': 'glove.6B.zip',  # Destination for GloVe embeddings
    },
}


def create_folder_structure():
    """Creates required folder structure."""
    folders = [
        DOWNLOAD_CONFIG['downloads_folder'],
        DOWNLOAD_CONFIG['datasets_folder'],
        os.path.join(DOWNLOAD_CONFIG['datasets_folder'], 'flickr8k'),
        os.path.join(DOWNLOAD_CONFIG['datasets_folder'], 'flickr30k'),
        os.path.join(DOWNLOAD_CONFIG['datasets_folder'], 'coco'),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Created directory: {folder}")
        print(f"Created directory: {folder}")


# Download datasets
def download_file(dataset_name, url, destination):
    """Downloads a single file with progress tracking."""
    if os.path.exists(destination):
        logging.info(f"File {destination} already exists, skipping download.")
        print(f"File {destination} already exists, skipping download.")
        return

    try:
        print(f"Downloading {dataset_name} from {url} to {destination}...")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:  # Filter out keep-alive chunks
                    downloaded += len(chunk)
                    f.write(chunk)
                    done = int(50 * downloaded / total_size)
                    print(f"\rProgress: [{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes", end='')

        print("\nDownload complete!")
        logging.info(f"Successfully downloaded {dataset_name}.")
    except Exception as e:
        logging.error(f"Failed to download {dataset_name}: {e}")
        print(f"Failed to download {dataset_name}: {e}")


def download_datasets():
    """Download datasets using multithreading."""
    create_folder_structure()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for dataset_name, url in DOWNLOAD_CONFIG['download_urls'].items():
            destination = os.path.join(
                DOWNLOAD_CONFIG['downloads_folder'],
                DOWNLOAD_CONFIG['download_destinations'][dataset_name]
            )
            executor.submit(download_file, dataset_name, url, destination)


# Extract zip files
def extract_zip_file(zip_path, dest_dir, specific_files=None, force_extract=False):
    """Extracts a ZIP file with optional file filtering and force extraction."""
    if not os.path.exists(zip_path):
        logging.warning(f"ZIP file {zip_path} not found. Skipping extraction.")
        print(f"ZIP file {zip_path} not found. Skipping extraction.")
        return

    if os.path.exists(dest_dir) and not force_extract:
        logging.info(f"Files already exist in {dest_dir}, skipping extraction.")
        print(f"Files already exist in {dest_dir}, skipping extraction.")
        return

    try:
        print(f"Extracting {zip_path} to {dest_dir} (Force: {force_extract})...")
        logging.info(f"Extracting {zip_path} to {dest_dir} (Force: {force_extract})...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if specific_files:
                for file in specific_files:
                    zip_ref.extract(file, dest_dir)
                    print(f"Extracted: {file}")
            else:
                zip_ref.extractall(dest_dir)
                print(f"Extracted all files to {dest_dir}")

        logging.info(f"Successfully extracted {zip_path}.")
    except Exception as e:
        logging.error(f"Error extracting {zip_path}: {e}")
        print(f"Error extracting {zip_path}: {e}")


def extract_caption_data(force_extract=False):
    """Ensure caption datasets like dataset_flickr8k.json are extracted."""
    zip_path = os.path.join(
        DOWNLOAD_CONFIG['downloads_folder'],
        DOWNLOAD_CONFIG['download_destinations']['ak_captions']
    )
    dest_dir = os.path.join(DOWNLOAD_CONFIG['datasets_folder'], 'flickr8k')
    specific_files = ['dataset_flickr8k.json']

    extract_zip_file(zip_path, dest_dir, specific_files, force_extract)


def extract_glove_embeddings(force_extract=False):
    """Extract GloVe embeddings to the datasets folder."""
    zip_path = os.path.join(
        DOWNLOAD_CONFIG['downloads_folder'],
        DOWNLOAD_CONFIG['download_destinations']['glove']
    )
    dest_dir = os.path.join(DOWNLOAD_CONFIG['datasets_folder'], 'glove')

    extract_zip_file(zip_path, dest_dir, specific_files=None, force_extract=force_extract)


# Main execution
if __name__ == "__main__":
    create_folder_structure()
    download_datasets()

    # Extract captions
    extract_caption_data(force_extract=True)

    # Extract GloVe embeddings
    extract_glove_embeddings(force_extract=True)
