#!/usr/bin/env python3
"""
Fixed Chinese MNIST Dataset Loader

This module provides functions to load the Chinese MNIST dataset from local files.
The dataset contains 15,000 handwritten Chinese numerals (0-9) with corresponding images.

Fixed version with correct filename mapping.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def load_chinese_mnist(
    data_dir: str = "chinese-mnist",
    max_samples: Optional[int] = None,
    target_size: Tuple[int, int] = (28, 28),
    normalize: bool = True,
    return_metadata: bool = False,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Load Chinese MNIST dataset from local files.
    
    Parameters:
    -----------
    data_dir : str
        Directory path containing the Chinese MNIST dataset
    max_samples : int, optional
        Maximum number of samples to load (for memory efficiency)
    target_size : tuple
        Target size for images (height, width)
    normalize : bool
        Whether to normalize pixel values to [0, 1]
    return_metadata : bool
        Whether to return metadata DataFrame
    random_state : int
        Random seed for sampling
        
    Returns:
    --------
    X : np.ndarray
        Image data of shape (n_samples, height*width)
    y : np.ndarray
        Labels (0-9 corresponding to Chinese numerals, mapped from 1-10 to 0-9)
    metadata : pd.DataFrame, optional
        Metadata if return_metadata=True
    """
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found")
    
    csv_path = os.path.join(data_dir, "chinese_mnist.csv")
    images_dir = os.path.join(data_dir, "data", "data")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    print(f"Loading Chinese MNIST dataset from: {data_dir}")
    
    # Load metadata
    print("Reading metadata...")
    metadata_df = pd.read_csv(csv_path)
    print(f"Found {len(metadata_df)} samples in metadata")
    
    # Sample subset if requested
    if max_samples and max_samples < len(metadata_df):
        np.random.seed(random_state)
        sample_indices = np.random.choice(
            len(metadata_df), size=max_samples, replace=False
        )
        metadata_df = metadata_df.iloc[sample_indices].reset_index(drop=True)
        print(f"Using random subset of {max_samples} samples")
    
    # Create correct filename mapping: input_{suite_id}_{code}_{sample_id}.jpg
    def get_filename(row):
        return f"input_{row['suite_id']}_{row['code']}_{row['sample_id']}.jpg"
    
    metadata_df['filename'] = metadata_df.apply(get_filename, axis=1)
    
    # Map labels from 1-10 to 0-9 for standard ML format
    # Value 10 (十) becomes 0, values 1-9 stay the same
    def map_labels(value):
        if value == 10:
            return 0  # Chinese "十" (10) -> 0
        else:
            return value  # 1-9 stay as 1-9
    
    metadata_df['mapped_label'] = metadata_df['value'].apply(map_labels)
    
    # Load images
    print("Loading images...")
    images = []
    labels = []
    valid_indices = []
    failed_count = 0
    
    for idx, row in metadata_df.iterrows():
        img_path = os.path.join(images_dir, row['filename'])
        
        if os.path.exists(img_path):
            try:
                # Load and process image
                img = Image.open(img_path)
                
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to target size
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Flatten to 1D
                img_flat = img_array.flatten()
                
                images.append(img_flat)
                labels.append(row['mapped_label'])
                valid_indices.append(idx)
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Only show first 5 errors
                    print(f"Warning: Could not load image {row['filename']}: {e}")
                continue
        else:
            failed_count += 1
            if failed_count <= 5:  # Only show first 5 missing files
                print(f"Warning: Image not found: {row['filename']}")
    
    if failed_count > 5:
        print(f"... and {failed_count - 5} more issues (not shown)")
    
    if len(images) == 0:
        raise ValueError("No valid images found!")
    
    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Normalize if requested
    if normalize:
        X = X / 255.0
        print("Pixel values normalized to [0, 1]")
    
    # Filter metadata to valid samples
    if return_metadata:
        metadata_final = metadata_df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"Successfully loaded {len(X)} images")
    print(f"Image shape: {target_size} -> flattened to {X.shape[1]} features")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if return_metadata:
        return X, y, metadata_final
    else:
        return X, y


def load_chinese_mnist_sample(n_samples=5000, **kwargs):
    """
    Convenience function to load a sample of Chinese MNIST for quick experiments.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to load
    **kwargs : dict
        Additional arguments passed to load_chinese_mnist
        
    Returns:
    --------
    X, y : numpy arrays
        Images and labels
    """
    return load_chinese_mnist(max_samples=n_samples, **kwargs)


def visualize_chinese_mnist_samples(X, y, n_samples=20, figsize=(15, 3)):
    """
    Visualize sample images from Chinese MNIST dataset.
    
    Parameters:
    -----------
    X : np.ndarray
        Image data
    y : np.ndarray
        Labels (0-9)
    n_samples : int
        Number of samples to display
    figsize : tuple
        Figure size
    """
    # Get Chinese character mapping
    char_mapping = get_chinese_character_mapping()
    
    # Determine grid size
    cols = min(n_samples, 10)
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Calculate image dimensions
    img_size = int(np.sqrt(X.shape[1]))
    
    for i in range(n_samples):
        if i < len(X):
            img = X[i].reshape(img_size, img_size)
            axes[i].imshow(img, cmap='gray')
            chinese_char = char_mapping.get(y[i], str(y[i]))
            axes[i].set_title(f'{y[i]}: {chinese_char}', fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Chinese MNIST Sample Images', y=1.02, fontsize=14)
    plt.show()


def get_chinese_character_mapping():
    """
    Get mapping from numeric labels (0-9) to Chinese characters.
    
    Returns:
    --------
    dict : Mapping from label (0-9) to Chinese character
    """
    return {
        0: '十',  # 10 -> mapped to 0
        1: '一',  # 1
        2: '二',  # 2  
        3: '三',  # 3
        4: '四',  # 4
        5: '五',  # 5
        6: '六',  # 6
        7: '七',  # 7
        8: '八',  # 8
        9: '九',  # 9
    }


def analyze_dataset_distribution(y, show_plot=True):
    """
    Analyze the distribution of labels in the dataset.
    
    Parameters:
    -----------
    y : np.ndarray
        Labels
    show_plot : bool
        Whether to show a bar plot
    """
    char_mapping = get_chinese_character_mapping()
    
    unique_labels, counts = np.unique(y, return_counts=True)
    
    print("Dataset Distribution:")
    print("-" * 30)
    for label, count in zip(unique_labels, counts):
        char = char_mapping.get(label, str(label))
        percentage = count / len(y) * 100
        print(f"Label {label} ({char}): {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Unique labels: {len(unique_labels)}")
    
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create labels with Chinese characters
        labels_with_chars = [f"{label}\n{char_mapping.get(label, str(label))}" 
                           for label in unique_labels]
        
        bars = ax.bar(range(len(unique_labels)), counts, alpha=0.7, color='skyblue')
        ax.set_xlabel('Chinese Numeral')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Chinese MNIST Dataset Distribution')
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(labels_with_chars)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   str(count), ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def save_sample_images(X, y, output_dir="chinese_mnist_samples", n_per_class=3):
    """
    Save sample images for each class to disk.
    
    Parameters:
    -----------
    X : np.ndarray
        Image data
    y : np.ndarray
        Labels
    output_dir : str
        Directory to save images
    n_per_class : int
        Number of samples per class to save
    """
    os.makedirs(output_dir, exist_ok=True)
    char_mapping = get_chinese_character_mapping()
    img_size = int(np.sqrt(X.shape[1]))
    
    for label in np.unique(y):
        # Find samples for this label
        indices = np.where(y == label)[0][:n_per_class]
        char = char_mapping.get(label, str(label))
        
        for i, idx in enumerate(indices):
            img = X[idx].reshape(img_size, img_size)
            # Convert back to 0-255 range
            img_uint8 = (img * 255).astype(np.uint8)
            
            filename = f"digit_{label}_{char}_sample_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            
            Image.fromarray(img_uint8, mode='L').save(filepath)
    
    print(f"Sample images saved to {output_dir}/")


# Test function
def test_loader():
    """Test the Chinese MNIST loader."""
    print("Testing Fixed Chinese MNIST loader...")
    
    try:
        # Test loading with different parameters
        X, y = load_chinese_mnist(max_samples=1000)
        print(f"✓ Successfully loaded {len(X)} samples")
        print(f"✓ Image shape: {X.shape}")
        print(f"✓ Labels shape: {y.shape}")
        print(f"✓ Label range: {y.min()}-{y.max()}")
        print(f"✓ Unique labels: {np.unique(y)}")
        
        # Analyze distribution
        analyze_dataset_distribution(y)
        
        # Test visualization
        print("\n✓ Creating visualization...")
        visualize_chinese_mnist_samples(X, y, n_samples=10)
        
        # Test with metadata
        X_meta, y_meta, metadata = load_chinese_mnist(
            max_samples=100, return_metadata=True
        )
        print(f"✓ Metadata loading successful: {metadata.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    test_loader()