#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Analysis Tab - Global analysis functions for entire videos

@author: standarduser
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm


def compute_mean_fft(frames, progress=gr.Progress()):
    """Computes mean FFT across all video frames"""
    if not frames or len(frames) == 0:
        return None, "No video loaded"
    
    progress(0, desc="Starting Mean FFT calculation...")
    
    # Initialize accumulator for FFT magnitudes
    first_frame = frames[0]
    if len(first_frame.shape) == 3:
        gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = first_frame
    
    h, w = gray.shape
    fft_accumulator = np.zeros((h, w), dtype=np.float64)
    
    # Process each frame
    total_frames = len(frames)
    for i, frame in enumerate(frames):
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Compute FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Magnitude spectrum (log scale)
        magnitude_spectrum = np.abs(f_shift)
        magnitude_spectrum = np.log1p(magnitude_spectrum)
        
        # Accumulate
        fft_accumulator += magnitude_spectrum
        
        # Update progress
        if i % 10 == 0:  # Update every 10 frames
            progress((i + 1) / total_frames, desc=f"Processing frame {i+1}/{total_frames}")
    
    # Compute mean
    mean_fft = fft_accumulator / total_frames
    
    progress(1.0, desc="Applying colormap...")
    
    # Normalize to 0-1
    if np.max(mean_fft) > 0:
        mean_fft_norm = mean_fft / np.max(mean_fft)
    else:
        mean_fft_norm = np.zeros_like(mean_fft)
    
    # Apply viridis colormap
    viridis = cm.get_cmap('viridis')
    colored = viridis(mean_fft_norm)
    
    # Convert to RGB and scale to 0-255
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    result_image = Image.fromarray(rgb)
    
    status_text = f"✓ Mean FFT calculated from {total_frames} frames"
    
    return result_image, status_text


def create_tab_video_analysis(tab_label):
    """Creates a tab for video-level analysis"""
    with gr.TabItem(tab_label):
        # State for video frames (will be shared from main tab)
        video_frames = gr.State([])
        mean_fft_result = gr.State(None)
        
        gr.Markdown("# Video Analysis")
        gr.Markdown("Global analysis functions that process the entire video")
        
        with gr.Accordion("Mean FFT Analysis", open=True):
            gr.Markdown("""
            **Purpose:** Calculate the mean FFT (Fast Fourier Transform) across all video frames.
            
            This analysis helps detect:
            - Consistent frequency patterns in AI-generated videos
            - Generator-specific artifacts
            - Periodic noise or compression artifacts
            
            *Note: This operation processes all frames and may take some time for long videos.*
            """)
            
            with gr.Row():
                btn_calculate_mean_fft = gr.Button(
                    "🔬 Calculate Mean FFT",
                    variant="primary",
                    size="lg",
                    scale=1
                )
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready - Load a video first",
                    interactive=False,
                    scale=2
                )
            
            with gr.Row():
                mean_fft_image = gr.Image(
                    label="Mean FFT Spectrum",
                    type="pil",
                    height=500,
                    interactive=False,    # Read-only
                    sources=[]           # No upload sources
                )
            
            with gr.Row():
                btn_download = gr.Button("💾 Download Result", size="sm")
        
        # Placeholder for future analysis tools
        with gr.Accordion("Future Analysis Tools", open=False):
            gr.Markdown("""
            **Coming soon:**
            - Temporal Consistency Analysis
            - Motion Pattern Detection
            - Frame-to-Frame Similarity Analysis
            - Audio-Video Synchronization Check
            """)
        
        # Event handlers
        btn_calculate_mean_fft.click(
            fn=compute_mean_fft,
            inputs=[video_frames],
            outputs=[mean_fft_image, status_text]
        )
        
        # Download functionality (triggers browser download)
        # Note: In Gradio, the image component already has download capability
        btn_download.click(
            fn=lambda img: img,
            inputs=[mean_fft_image],
            outputs=[mean_fft_image]
        )
        
        # Return the video_frames state so it can be connected from the main tab
        return video_frames
