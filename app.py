#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVAT - Synthetic Video Analyze Tool
Created on Wed Oct 29 12:00:23 2025
@author: standarduser
"""

import gradio as gr
from tabs.tab_videoframes import create_tab_videoframes
from tabs.tab_info import create_tab_info
from tabs.tab_video_analysis import create_tab_video_analysis
from tabs.tab_classify_image import create_tab_classify_image
from processing.image_processing import process_image

# Gradio App erstellen
with gr.Blocks() as demo:
    gr.Markdown("# SVAT - Synthetic Video Analyze Tool")
    gr.Markdown("*Analyze videos for synthetic/AI-generated content artifacts*")
    
    # Shared state for video frames across tabs
    shared_video_frames = gr.State([])
    
    with gr.Tabs():
        # Tab 1: Frame-by-frame analysis
        video_frames_output = create_tab_videoframes("Video-Frames", process_image, shared_video_frames)
        
        # Tab 2: Video-level analysis
        video_analysis_frames = create_tab_video_analysis("Video Analysis")
       
        create_tab_classify_image("Classify Image") 
       
        # Tab 4: Help
        create_tab_info("Info")
    
    # Connect the video frames state between tabs
    # When frames are loaded in tab 1, update tab 2
    video_frames_output.change(
        fn=lambda frames: frames,
        inputs=[video_frames_output],
        outputs=[video_analysis_frames]
    )


if __name__ == "__main__":
    demo.launch()