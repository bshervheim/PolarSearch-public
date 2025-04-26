import os
import re
import torch
import numpy as np
import requests
import urllib.request
from PIL import Image, UnidentifiedImageError, ImageFile
from io import BytesIO
import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_robust_session():
    """
    Create a robust session for downloading images with retries
    
    Returns:
        requests.Session: Configured session with retry strategy
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session

def fix_problematic_image(image_data):
    """
    Attempt to fix and load problematic image data
    
    Args:
        image_data (bytes): Raw image data
        
    Returns:
        PIL.Image: Image object or random noise if loading fails
    """
    # First try to open directly
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception:
        pass

    # Try to save and reload as PNG
    try:
        temp_buffer = BytesIO()
        temp_buffer.write(image_data)
        temp_buffer.seek(0)

        # Try different formats for reading
        for fmt in ['PNG', 'JPEG', 'GIF', 'BMP', 'WEBP']:
            temp_buffer.seek(0)
            try:
                img = Image.open(temp_buffer, formats=[fmt])
                img = img.convert('RGB')
                output_buffer = BytesIO()
                img.save(output_buffer, format='PNG')
                output_buffer.seek(0)
                return Image.open(output_buffer)
            except:
                continue
    except:
        pass

    # If all else fails, create a random noise image
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

def preload_images(urls, cache_dir):
    """
    Preload images in parallel with caching for faster training
    
    Args:
        urls (list): List of image URLs to download
        cache_dir (str): Directory to store cached images
        
    Returns:
        dict: Mapping of indices to cached image paths and success status
    """
    os.makedirs(cache_dir, exist_ok=True)
    session = create_robust_session()

    def download_image(idx_url):
        idx, url = idx_url
        try:
            # Clean URL - replace spaces with %20
            url = url.replace(' ', '%20')
            
            # Create a safe filename
            pattern = r'[^a-zA-Z0-9\-_.]'
            basename = os.path.basename(url)
            safe_basename = re.sub(pattern, '_', basename)
            safe_filename = f"{idx}_{safe_basename}"
            cache_path = os.path.join(cache_dir, safe_filename)

            if os.path.exists(cache_path):
                return idx, cache_path, True

            response = session.get(url, timeout=5)
            if response.status_code == 200:
                # Try to fix problematic images
                try:
                    image = fix_problematic_image(response.content)
                    image.save(cache_path, format='PNG')
                    return idx, cache_path, True
                except Exception:
                    # If fixing fails, still save the original content
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                    return idx, cache_path, True
        except Exception:
            return idx, None, False

    print(f"Preloading {len(urls)} images...")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_url = {executor.submit(download_image, (i, url)): i for i, url in enumerate(urls)}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            idx, path, success = future.result()
            results[idx] = (path, success)
            if i % 100 == 0:
                print(f"Preloaded {i}/{len(urls)} images")

    success_count = sum(1 for _, success in results.values() if success)
    print(f"Successfully preloaded {success_count}/{len(urls)} images ({success_count/len(urls)*100:.1f}%)")
    return results

class ImageSetDataset(Dataset):
    """
    Dataset for loading images grouped by sets with error handling
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with image_url and set_id columns
        transform (torchvision.transforms): Image transformations
        cache_dir (str): Directory to cache downloaded images
        preload (bool): Whether to preload all images at initialization
    """
    def __init__(self, dataframe, transform=None, cache_dir='cached_images', preload=True):
        self.df = dataframe
        self.transform = transform
        self.cache_dir = cache_dir

        # Group images by set_id
        self.sets = defaultdict(list)
        for _, row in self.df.iterrows():
            self.sets[row['set_id']].append(row)

        # Create flat list of (image_url, set_id) pairs
        self.image_data = []
        for set_id, images in self.sets.items():
            for img_data in images:
                self.image_data.append((img_data['image_url'], set_id))

        print(f"Dataset loaded with {len(self.image_data)} images across {len(self.sets)} sets")

        # Preload images if requested
        if preload:
            self.preloaded = preload_images([url for url, _ in self.image_data], cache_dir)
        else:
            self.preloaded = None

        # Statistics
        self.success_count = 0
        self.error_count = 0

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_url, set_id = self.image_data[idx]

        # If we have preloaded images, use them
        if self.preloaded is not None:
            cache_path, success = self.preloaded.get(idx, (None, False))

            if success and cache_path:
                try:
                    # Try to open with special handling for problematic images
                    with open(cache_path, 'rb') as f:
                        image_data = f.read()

                    image = fix_problematic_image(image_data)
                    if self.transform:
                        image = self.transform(image)
                    self.success_count += 1
                    return image, set_id
                except Exception:
                    # Fall back to direct loading if cache fails
                    pass

        # Direct loading path
        try:
            # Clean the URL - replace spaces with %20
            img_url = img_url.replace(' ', '%20')

            # Try to download and load the image
            request = urllib.request.Request(
                img_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                image_data = response.read()

            # Try with our fix function
            image = fix_problematic_image(image_data)

            if self.transform:
                image = self.transform(image)

            self.success_count += 1
            return image, set_id

        except Exception:
            self.error_count += 1
            # Create a random noise image instead of black
            if self.transform:
                # Create a random tensor directly
                image = torch.rand(3, 224, 224)
                # Normalize to match the transform
                image = (image - 0.5) / 0.5
            else:
                # Create a random PIL image
                random_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(random_array)
                if self.transform:
                    image = self.transform(image)

            return image, set_id

def get_transforms():
    """
    Get standard transforms for ViT models
    
    Returns:
        torchvision.transforms: Composition of image transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def collate_with_set_labels(batch):
    """
    Custom collate function that converts set_ids to numerical labels
    
    Args:
        batch (list): Batch of (image, set_id) pairs
        
    Returns:
        tuple: (images_tensor, labels_tensor)
    """
    images, set_ids = zip(*batch)
    images = torch.stack(images)

    # Convert set_ids to numerical labels
    unique_set_ids = sorted(list(set(set_ids)))  # Sort for determinism
    set_id_to_idx = {set_id: idx for idx, set_id in enumerate(unique_set_ids)}
    labels = torch.tensor([set_id_to_idx[set_id] for set_id in set_ids])

    return images, labels

def create_dataloader(dataset, batch_size=32, num_workers=4):
    """
    Create an optimized DataLoader for the dataset
    
    Args:
        dataset (Dataset): Dataset to load from
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: Configured data loader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_set_labels,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
