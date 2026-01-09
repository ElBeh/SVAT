# image_processing.py
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
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
    
    # Normalize to 0-255
    laplacian_norm = np.absolute(laplacian)
    laplacian_norm = np.uint8(255 * laplacian_norm / np.max(laplacian_norm)) if np.max(laplacian_norm) > 0 else np.uint8(laplacian_norm)
    
    return Image.fromarray(laplacian_norm)

def fft_spectrum(img):
    """Computes 2D FFT and visualizes log-scaled magnitude spectrum."""
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
    
    # Normalize to 0-255
    magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
    
    return Image.fromarray(magnitude_spectrum)

def error_level_analysis(img, quality=90):
    """Performs Error Level Analysis via JPEG re-compression."""
    arr = np.array(img)
    
    # Save with specified quality
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format='JPEG', quality=quality)
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
    """Decomposes image into wavelet subbands (LL, LH, HL, HH)."""
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
    
    # Normalize each subband
    def normalize(band):
        band = np.abs(band)
        if np.max(band) > 0:
            return np.uint8(255 * band / np.max(band))
        return np.uint8(band)
    
    LL_norm = normalize(LL)
    LH_norm = normalize(LH)
    HL_norm = normalize(HL)
    HH_norm = normalize(HH)
    
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
    
    # Normalize
    magnitude = np.uint8(255 * magnitude / np.max(magnitude)) if np.max(magnitude) > 0 else np.uint8(magnitude)
    
    return Image.fromarray(magnitude)

def histogram_stretching(img):
    """Applies extreme contrast stretching."""
    arr = np.array(img)
    
    # Process each channel separately
    if len(arr.shape) == 3:
        result = np.zeros_like(arr)
        for i in range(arr.shape[2]):
            channel = arr[:, :, i]
            # Stretch to full range
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                stretched = 255 * (channel - min_val) / (max_val - min_val)
                result[:, :, i] = np.uint8(stretched)
            else:
                result[:, :, i] = channel
        return Image.fromarray(result)
    else:
        # Grayscale
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            stretched = 255 * (arr - min_val) / (max_val - min_val)
            return Image.fromarray(np.uint8(stretched))
        return Image.fromarray(arr)

def process_image(slider_input, transformation):
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
        "Error Level Analysis": error_level_analysis,
        "Wavelet Decomposition": wavelet_decomposition,
        "Noise Extraction": noise_extraction,
        "YCbCr Channels": ycbcr_channels,
        "Gradient Magnitude": gradient_magnitude,
        "Histogram Stretching": histogram_stretching
    }
    
    transform_func = transform_functions.get(transformation, laplacian_highpass)
    transformed = transform_func(img)
    
    # Return as tuple for ImageSlider
    return (img, transformed)