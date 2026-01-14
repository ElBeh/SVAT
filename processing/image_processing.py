# image_processing.py
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

def laplacian_highpass(img):
    """Applies Laplacian high-pass filter to emphasize high frequencies."""
    arr = np.array(img)
    
    # Convert to grayscale if needed
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    
    # Normalize to 0-255 (fixed for NumPy 2.x)
    laplacian_norm = np.absolute(laplacian)
    if np.max(laplacian_norm) > 0:
        laplacian_norm = (laplacian_norm / np.max(laplacian_norm)) * 255.0
        laplacian_norm = np.clip(laplacian_norm, 0, 255).astype(np.uint8)
    else:
        laplacian_norm = np.zeros_like(laplacian_norm, dtype=np.uint8)
    
    return Image.fromarray(laplacian_norm)

def fft_spectrum(img):
    """Computes 2D FFT and visualizes log-scaled magnitude spectrum with viridis colormap."""
    arr = np.array(img)
    
    # Convert to grayscale
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    
    # Compute FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Magnitude spectrum (log scale)
    magnitude_spectrum = np.abs(f_shift)
    magnitude_spectrum = np.log1p(magnitude_spectrum)
    
    # Normalize to 0-1 (fixed for NumPy 2.x)
    if np.max(magnitude_spectrum) > 0:
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    else:
        magnitude_spectrum = np.zeros_like(magnitude_spectrum)
    
    # Apply viridis colormap (blue-green-yellow)
    viridis = cm.get_cmap('viridis')
    colored = viridis(magnitude_spectrum)  # Returns RGBA in range 0-1
    
    # Convert to RGB (remove alpha channel) and scale to 0-255
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(rgb)

def error_level_analysis(img, quality=90):
    """Performs Error Level Analysis via JPEG re-compression."""
    arr = np.array(img)
    
    # Save with specified quality
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format='JPEG', quality=int(quality))
    buffer.seek(0)
    
    # Reload compressed image
    compressed_img = Image.open(buffer)
    compressed_arr = np.array(compressed_img)
    
    # Compute difference
    diff = cv2.absdiff(arr, compressed_arr)
    
    # Enhance differences
    diff = cv2.multiply(diff, 10)
    
    # Convert to grayscale if color
    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    return Image.fromarray(diff)

def wavelet_decomposition(img):
    """Decomposes image into wavelet subbands with proper visualization"""
    if not PYWT_AVAILABLE:
        # Fallback: return grayscale
        arr = np.array(img)
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        return Image.fromarray(gray)
    
    arr = np.array(img)
    
    # Convert to grayscale
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    
    # Perform 2D wavelet decomposition
    coeffs = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Different normalization for LL vs high-frequency bands
    def normalize_lowfreq(band):
        """Standard normalization for LL (approximation)"""
        if np.max(band) > 0:
            band = (band / np.max(band)) * 255.0
            return np.clip(band, 0, 255).astype(np.uint8)
        return np.zeros_like(band, dtype=np.uint8)
    
    def normalize_highfreq(band, amplification=30):
        """Amplified normalization for high-frequency details"""
        band = np.abs(band)
        
        # Amplify before normalization
        band = band * amplification
        
        # Clip to prevent overflow
        band = np.clip(band, 0, 255)
        
        # Normalize to full range
        if np.max(band) > 0:
            band = (band / np.max(band)) * 255.0
            return band.astype(np.uint8)
        return np.zeros_like(band, dtype=np.uint8)
    
    LL_norm = normalize_lowfreq(LL)
    LH_norm = normalize_highfreq(LH)  # Horizontal edges
    HL_norm = normalize_highfreq(HL)  # Vertical edges
    HH_norm = normalize_highfreq(HH)  # Diagonal edges/noise
    
    # Combine into single image (2x2 grid)
    top = np.hstack([LL_norm, LH_norm])
    bottom = np.hstack([HL_norm, HH_norm])
    combined = np.vstack([top, bottom])
    
    return Image.fromarray(combined)

def noise_extraction(img):
    """Extracts and amplifies noise via high-pass filter."""
    arr = np.array(img)
    
    # Convert to grayscale
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    
    # Apply strong Gaussian blur
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Subtract to get high-frequency content (noise)
    noise = cv2.subtract(gray, blurred)
    
    # Amplify noise
    noise = cv2.multiply(noise, 5)
    
    return Image.fromarray(noise)

def ycbcr_channels(img):
    """Converts to YCbCr and visualizes chrominance channels."""
    arr = np.array(img)
    
    # Convert RGB to YCbCr
    if len(arr.shape) == 3:
        ycbcr = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        
        # Extract channels
        Y, Cr, Cb = cv2.split(ycbcr)
        
        # Create combined visualization (Y on top, Cr and Cb side by side below)
        h, w = Y.shape
        
        # Resize Cr and Cb to half width
        Cr_resized = cv2.resize(Cr, (w//2, h//2))
        Cb_resized = cv2.resize(Cb, (w//2, h//2))
        
        # Combine
        bottom = np.hstack([Cr_resized, Cb_resized])
        Y_resized = cv2.resize(Y, (w, h//2))
        combined = np.vstack([Y_resized, bottom])
        
        return Image.fromarray(combined)
    else:
        return Image.fromarray(arr)

def gradient_magnitude(img):
    """Computes gradient magnitude using Sobel operator."""
    arr = np.array(img)
    
    # Convert to grayscale
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255 (fixed for NumPy 2.x)
    if np.max(magnitude) > 0:
        magnitude = (magnitude / np.max(magnitude)) * 255.0
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    else:
        magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    
    return Image.fromarray(magnitude)

def histogram_stretching(img):
    """Applies CLAHE for adaptive contrast enhancement"""
    arr = np.array(img)
    
    if len(arr.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        # Merge and convert back
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)
    else:
        # For grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        result = clahe.apply(arr)
        return Image.fromarray(result)

def process_image(slider_input, transformation, quality=90):
    """Applies the selected transformation."""
    # Extract image from slider input
    if slider_input is None:
        return None
    
    # If it's a tuple, take the first image
    if isinstance(slider_input, tuple):
        img = slider_input[0]
    else:
        img = slider_input
    
    if img is None:
        return None
    
    # Select the corresponding function
    transform_functions = {
        "Laplacian High-Pass": laplacian_highpass,
        "FFT Spectrum": fft_spectrum,
        "Error Level Analysis": lambda img: error_level_analysis(img, quality),
        "Wavelet Decomposition": wavelet_decomposition,
        "Noise Extraction": noise_extraction,
        "YCbCr Channels": ycbcr_channels,
        "Gradient Magnitude": gradient_magnitude,
        "Histogram Stretching": histogram_stretching,
        "None": lambda img: img  # Add None transformation
    }
    
    transform_func = transform_functions.get(transformation)
    if transform_func is None:
        return (img, img)
    
    transformed = transform_func(img)
    
    # Return as tuple for ImageSlider
    return (img, transformed)