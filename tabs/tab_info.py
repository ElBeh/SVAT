#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Help Tab
Created on Sat Nov  8 11:58:29 2025
@author: standarduser
"""

import gradio as gr
        
        
def create_tab_info(tab_label):
    """Creates a tab for help text"""
    with gr.TabItem(tab_label):
        gr.Markdown("""
# SVAT - Synthetic Video Analyze Tool
## Quick Start
1. **Load Video:** Go to "Video-Frames" tab and upload a video
2. **Navigate Frames:** Use slider or buttons to move through frames
3. **Apply Transformations:** Click transformation buttons to analyze frames
4. **Annotate:** Draw on frames in "Annotations" tab
5. **Analyze Video:** Use "Video Analysis" tab for global analysis
## Frame Transformations
### Laplacian High-Pass
Emphasizes high-frequency details and edges. Useful for detecting sharpness artifacts.
### FFT Spectrum
Shows frequency domain representation with viridis colormap (blue-green-yellow).
Reveals periodic patterns and compression artifacts.
### Error Level Analysis (ELA)
Detects JPEG compression artifacts by re-compressing the image.
Lower quality = more visible differences in manipulated areas.
### Wavelet Decomposition
Multi-scale frequency analysis showing LL, LH, HL, HH subbands.
Reveals different frequency components.
### Noise Extraction
Isolates high-frequency noise via high-pass filtering.
Shows noise patterns that might indicate generation artifacts.
### YCbCr Channels
Separates luminance (Y) and chrominance (Cb, Cr) channels.
Useful for detecting color space artifacts.
### Gradient Magnitude
Visualizes edge strength using Sobel operator.
Shows edge consistency.
### Histogram Stretching (CLAHE)
Adaptive contrast enhancement that preserves local details.
## Video Analysis
### Mean FFT
Calculates average FFT across all frames to detect:
- Consistent frequency patterns in AI-generated videos
- Generator-specific fingerprints
- Temporal artifacts
## Annotation Modes
**Per Frame (A):** Separate drawings for each frame  
**Global (B):** One drawing overlaid on all frames
## Tips for AI Detection
- Look for **repeating patterns** in FFT spectrum
- Check **ELA** for inconsistent compression levels
- Use **Mean FFT** to find generator fingerprints
- Compare **noise patterns** between frames
- Watch for **unnatural frequency distributions**
## Keyboard Shortcuts
*Navigation:*
- Use frame slider for quick navigation
- Click ◀/▶ buttons for precise frame control
## System Requirements
- Python 3.8+
- Gradio 6.x
- OpenCV
- NumPy 2.x
- Pillow
- Matplotlib
## About
SVAT is designed to help identify synthetic/AI-generated video content through various image analysis techniques.
Version: 0.5  
Updated: 
- 29.10.2025 Initial version
- 13.01.2026 added "Classify Image" Tab and classify function with XGBoost via image statistics
        """)