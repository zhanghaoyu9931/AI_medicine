import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import struct
from scipy import signal
import pandas as pd

def create_output_dir(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

def load_image(image_path: str) -> np.ndarray:
    """Load image from path."""
    return cv2.imread(image_path)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to enhance contrast."""
    return cv2.equalizeHist(image)

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8,8)) -> np.ndarray:
    """Enhance contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def gaussian_smoothing(image: np.ndarray, kernel_size: Tuple[int, int] = (5,5)) -> np.ndarray:
    """Apply Gaussian smoothing to reduce noise."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def process_single_image(image_path: str, output_path: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Process a single image with all preprocessing steps."""
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    image = convert_to_grayscale(image)

    # Apply preprocessing steps
    image = histogram_equalization(image)
    image = enhance_contrast(image)
    image = gaussian_smoothing(image)
    image = resize_image(image, target_size)

    # Save processed image
    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, image)

def process_all_images(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Process all images in the input directory."""
    create_output_dir(output_dir)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, output_dir, target_size)

def visualize_preprocessing(image_path: str) -> None:
    """Visualize preprocessing steps on a single image."""
    # Load original image
    original = load_image(image_path)
    if original is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    original = convert_to_grayscale(original)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')

    # Histogram equalization
    hist_eq = histogram_equalization(original)
    axes[0,1].imshow(hist_eq, cmap='gray')
    axes[0,1].set_title('Histogram Equalization')
    axes[0,1].axis('off')

    # Contrast enhancement
    contrast = enhance_contrast(original)
    axes[1,0].imshow(contrast, cmap='gray')
    axes[1,0].set_title('Contrast Enhancement')
    axes[1,0].axis('off')

    # Gaussian smoothing
    smoothed = gaussian_smoothing(original)
    axes[1,1].imshow(smoothed, cmap='gray')
    axes[1,1].set_title('Gaussian Smoothing')
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================================
# ECG Preprocessing Functions
# ============================================================================

def preprocess_jiachen_files(data_dir: str, labels_file: str) -> None:
    """Preprocess Jiachen's ECG files to fix naming inconsistencies"""
    data_path = Path(data_dir)
    labels_path = Path(labels_file)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_file}")
        return
    
    print("Preprocessing Jiachen's files...")
    
    # Find all .dat files with _H suffix
    dat_files_with_h = list(data_path.glob('*_H.dat'))
    print(f"Found {len(dat_files_with_h)} files with _H suffix")
    
    renamed_count = 0
    for dat_file in dat_files_with_h:
        # Create new filename without _H
        new_dat_name = dat_file.name.replace('_H.dat', '.dat')
        new_dat_path = dat_file.parent / new_dat_name
        
        # Rename .dat file if target doesn't exist
        if not new_dat_path.exists():
            dat_file.rename(new_dat_path)
            renamed_count += 1
            print(f"Renamed: {dat_file.name} -> {new_dat_name}")
    
    print(f"Renamed {renamed_count} .dat files")
    
    # Update labels.csv to remove _H suffix from IDs
    try:
        df = pd.read_csv(labels_file)
        if 'id' in df.columns:
            # Remove _H suffix from IDs
            df['id'] = df['id'].astype(str).str.replace('_H', '', regex=False)
            
            # Save updated labels
            df.to_csv(labels_file, index=False)
            print(f"Updated {labels_file} to remove _H suffix from IDs")
            print(f"Updated labels shape: {df.shape}")
            print(f"Sample IDs: {df['id'].head().tolist()}")
        else:
            print("Warning: 'id' column not found in labels file")
    except Exception as e:
        print(f"Error updating labels file: {e}")

def read_physionet_data(dat_file, hea_file):
    """Read PhysioNet format ECG data (.dat + .hea files)"""
    print(f"Reading ECG data from {Path(dat_file).name}")
    return read_physionet_manual(dat_file, hea_file)

def read_physionet_manual(dat_file, hea_file):
    """Manual parsing of PhysioNet format"""
    # Read header file
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    
    header_line = lines[0].strip().split()
    record_name = header_line[0]
    n_leads = int(header_line[1])
    sampling_rate = int(header_line[2])
    n_samples = int(header_line[3])
    
    print(f"Reading {record_name}: {n_leads} leads, {sampling_rate}Hz, {n_samples} samples")
    
    # Read binary data
    with open(dat_file, 'rb') as f:
        data = f.read()
    
    # Parse binary data (16-bit signed integers, interleaved by leads)
    n_values = len(data) // 2
    values = struct.unpack(f'<{n_values}h', data)
    
    try:
        expected_values = n_samples * n_leads
        if len(values) >= expected_values:
            values = values[:expected_values]
            signals = np.array(values).reshape(n_samples, n_leads)
        else:
            print(f"Warning: Expected {expected_values} values, got {len(values)}")
            padded_values = list(values) + [0] * (expected_values - len(values))
            signals = np.array(padded_values).reshape(n_samples, n_leads)
        
        return signals, sampling_rate, n_samples
    except Exception as e:
        print(f"Error reshaping data: {e}")
        return None, None, None

def process_ecg_signal(signals, sampling_rate, max_length=5000, start_threshold=500, end_threshold=150):
    """
    ECG processing with valid signal range detection and uniform sampling
    
    Args:
        signals: Raw ECG signals (samples, leads)
        sampling_rate: Original sampling rate
        max_length: Target length after uniform sampling
        start_threshold: Threshold to detect signal start (left side)
        end_threshold: Threshold to detect signal end (right side)
    
    Returns:
        processed_signal: 1D array of processed ECG signal
        valid_duration: Valid signal duration in seconds
        valid_range: Tuple of (start_idx, end_idx) for valid signal range
    """
    if signals is None:
        return None, None, None
    
    # Select first lead only
    if len(signals.shape) > 1:
        ecg_signal = signals[:, 0].astype(np.float32)
    else:
        ecg_signal = signals.astype(np.float32)
    
    # Find signal start with higher threshold
    start_indices = np.where(ecg_signal > start_threshold)[0]
    if len(start_indices) == 0:
        print(f"Warning: No signal above start threshold {start_threshold} found")
        return None, None, None
    
    # Find signal end with lower threshold
    end_indices = np.where(ecg_signal > end_threshold)[0]
    if len(end_indices) == 0:
        print(f"Warning: No signal above end threshold {end_threshold} found")
        return None, None, None
    
    # Get valid signal range
    start_idx = start_indices[0]  # First point > start_threshold
    end_idx = end_indices[-1]     # Last point > end_threshold
    valid_range = (start_idx, end_idx)
    
    # Extract valid signal segment
    valid_signal = ecg_signal[start_idx:end_idx+1]
    valid_length = len(valid_signal)
    valid_duration = valid_length / sampling_rate
    
    print(f"Valid signal range: [{start_idx}:{end_idx}] ({valid_length:,} samples, {valid_duration:.1f}s)")
    print(f"Original signal: {len(ecg_signal):,} samples, Valid signal: {valid_length:,} samples ({valid_length/len(ecg_signal)*100:.1f}%)")
    
    # Uniform sampling across valid signal range
    if valid_length > max_length:
        indices = np.linspace(0, valid_length - 1, max_length, dtype=int)
        processed_signal = valid_signal[indices]
    else:
        processed_signal = valid_signal
    
    # Remove DC offset only
    processed_signal = processed_signal - np.mean(processed_signal)
    
    return processed_signal, valid_duration, valid_range

def process_all_ecg_signals(input_dir: str, output_dir: str, max_length: int = 5000) -> int:
    """
    Process all ECG signals in the directory with uniform sampling
    
    Args:
        input_dir: Directory containing .dat and .hea files
        output_dir: Directory to save processed .npy files
        max_length: Target length for uniform sampling
    
    Returns:
        Number of successfully processed files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dat_files = list(input_path.glob('*.dat'))
    print(f"Processing {len(dat_files)} ECG files...")
    
    processed_count = 0
    
    for dat_file in dat_files:
        hea_file = Path(str(dat_file).replace('.dat', '.hea'))
        output_file = output_path / f"{dat_file.stem}.npy"
        
        # Skip if already processed
        if output_file.exists():
            continue
            
        if not hea_file.exists():
            print(f"✗ {dat_file.name}: header file missing")
            continue
        
        try:
            # Read ECG data
            raw_signals, sampling_rate, n_samples = read_physionet_data(dat_file, hea_file)
            
            if raw_signals is None:
                print(f"✗ {dat_file.name}: failed to read data")
                continue
            
            # Process signal
            processed_signal, valid_duration, valid_range = process_ecg_signal(
                raw_signals, sampling_rate, max_length
            )
            
            if processed_signal is None:
                print(f"✗ {dat_file.name}: failed to process")
                continue
            
            # Save processed signal
            np.save(output_file, processed_signal)
            processed_count += 1
            
            if processed_count <= 3:
                print(f"✓ {dat_file.name}: {len(processed_signal)} points, {valid_duration:.1f}s valid span")
                
        except Exception as e:
            print(f"✗ {dat_file.name}: {e}")
    
    if processed_count > 3:
        print(f"... and {processed_count - 3} more files processed")
    print(f"Total: {processed_count} ECG files processed")
    
    return processed_count

def visualize_ecg_preprocessing(dat_file, hea_file, max_length=5000, start_threshold=500, end_threshold=150):
    """Visualize ECG preprocessing: original vs valid range vs processed signal"""
    print(f"Visualizing: {Path(dat_file).name}")
    
    # Read raw data
    raw_signals, sampling_rate, n_samples = read_physionet_data(dat_file, hea_file)
    
    if raw_signals is None:
        print("Failed to read ECG data")
        return
    
    # Get original signal
    if len(raw_signals.shape) > 1:
        raw_complete = raw_signals[:, 0]  # First lead, complete signal
    else:
        raw_complete = raw_signals
    
    # Process signal (this will find valid range and process)
    processed_signal, valid_duration, valid_range = process_ecg_signal(
        raw_signals, sampling_rate, max_length, start_threshold, end_threshold
    )
    
    if processed_signal is None:
        print("Failed to process signal")
        return
    
    # Extract valid signal segment for comparison
    start_idx, end_idx = valid_range
    valid_signal = raw_complete[start_idx:end_idx+1]
    
    # Create time axes
    raw_time = np.arange(len(raw_complete)) / sampling_rate
    valid_time = np.arange(start_idx, end_idx+1) / sampling_rate
    processed_time = np.linspace(valid_time[0], valid_time[-1], len(processed_signal))
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Complete original signal with valid range highlighted
    print(f"Plotting complete original signal: {len(raw_complete):,} points...")
    axes[0].plot(raw_time, raw_complete, linewidth=0.3, color='gray', alpha=0.6, label='Complete signal')
    axes[0].plot(valid_time, valid_signal, linewidth=1.0, color='blue', label='Valid range')
    axes[0].axhline(y=start_threshold, color='red', linestyle='--', alpha=0.8, label=f'Start threshold (>{start_threshold})')
    axes[0].axhline(y=end_threshold, color='orange', linestyle='--', alpha=0.8, label=f'End threshold (>{end_threshold})')
    axes[0].set_title(f'Complete Original Signal ({len(raw_complete):,} points, {sampling_rate}Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    
    # Valid signal range only
    axes[1].plot(valid_time, valid_signal, linewidth=0.8, color='blue')
    axes[1].set_title(f'Valid Signal Range ({len(valid_signal):,} points, {valid_duration:.1f}s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    
    # Processed signal (uniform sampling from valid range)
    axes[2].plot(processed_time, processed_signal, linewidth=1.0, color='green')
    axes[2].set_title(f'Processed Signal ({len(processed_signal)} points, uniform sampling from valid range)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original: {len(raw_complete):,} samples at {sampling_rate}Hz")
    print(f"Valid range: {len(valid_signal):,} samples ({valid_duration:.1f}s, {len(valid_signal)/len(raw_complete)*100:.1f}%)")
    print(f"Processed: {len(processed_signal)} samples from valid range")
    print(f"Valid signal compression ratio: {len(valid_signal)/len(processed_signal):.1f}:1")

