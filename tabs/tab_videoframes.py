#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 09:54:54 2025

@author: standarduser
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image

# CSS for box styling
css = """
.box {
    border: 2px solid #4CAF50;
    padding: 10px;
    border-radius: 10px;
    background-color: #f9f9f9;
}
"""

def merge_annotations(base_image, annotations, mode, current_frame_idx, global_annotation):
    """Combines base frame with annotations"""
    if base_image is None:
        return None
    
    if isinstance(base_image, np.ndarray):
        img = Image.fromarray(base_image)
    else:
        img = base_image.copy()
    
    # Mode B: Global annotation
    if mode == "B" and global_annotation is not None:
        img = Image.alpha_composite(img.convert('RGBA'), global_annotation).convert('RGB')
    
    # Mode A: Frame-specific annotation
    elif mode == "A" and current_frame_idx in annotations:
        img = Image.alpha_composite(img.convert('RGBA'), annotations[current_frame_idx]).convert('RGB')
    
    return img

def apply_transformation(frame, transformation, process_image_func):
    """Applies selected transformation to frame"""
    if frame is None or transformation == "None":
        return frame
    
    # Convert numpy array to PIL if needed
    if isinstance(frame, np.ndarray):
        pil_frame = Image.fromarray(frame)
    else:
        pil_frame = frame
    
    # Call process_image with the frame
    result = process_image_func(pil_frame, transformation)
    
    # Extract transformed image from tuple
    if isinstance(result, tuple) and len(result) == 2:
        transformed = result[1]
    else:
        transformed = result
    
    # Convert back to numpy array
    if transformed is not None:
        return np.array(transformed)
    
    return frame

def create_sketchpad_value(base_image, annotations, mode, current_frame_idx, global_annotation, transformation, process_image_func):
    """Creates Sketchpad value (Background + Layers)"""
    if base_image is None:
        return None
    
    # Apply transformation first
    transformed_frame = apply_transformation(base_image, transformation, process_image_func)
    
    # Prepare base image
    if isinstance(transformed_frame, np.ndarray):
        background = Image.fromarray(transformed_frame)
    else:
        background = transformed_frame.copy()
    
    # Extract annotation layer
    annotation_layer = None
    if mode == "B" and global_annotation is not None:
        annotation_layer = global_annotation
    elif mode == "A" and current_frame_idx in annotations:
        annotation_layer = annotations[current_frame_idx]
    
    # Create Sketchpad dict
    result = {
        'background': background,
        'layers': [annotation_layer] if annotation_layer is not None else [],
        'composite': None
    }
    
    return result

def extract_annotation_from_sketch(sketch_data):
    """Extracts only the drawing from Sketchpad data"""
    if sketch_data is None:
        return None
    
    if isinstance(sketch_data, dict):
        if 'layers' in sketch_data and len(sketch_data['layers']) > 0:
            drawing = sketch_data['layers'][0]
            if isinstance(drawing, np.ndarray):
                # Check if there are actually drawings
                if len(drawing.shape) == 3 and drawing.shape[2] == 4:  # RGBA
                    alpha = drawing[:, :, 3]
                    if np.any(alpha > 0):
                        return Image.fromarray(drawing, 'RGBA')
                return None
            return drawing
        elif 'composite' in sketch_data and sketch_data['composite'] is not None:
            composite = sketch_data['composite']
            if isinstance(composite, np.ndarray):
                return Image.fromarray(composite, 'RGBA')
            return composite
    
    return None

def create_comparison_slider(frame, transformation, process_image_func):
    """Creates ImageSlider comparison between original and transformed frame"""
    if frame is None:
        return None
    
    # Convert to PIL if needed
    if isinstance(frame, np.ndarray):
        original = Image.fromarray(frame)
    else:
        original = frame
    
    if transformation == "None":
        return (original, original)
    
    # Apply transformation
    transformed_array = apply_transformation(frame, transformation, process_image_func)
    
    if isinstance(transformed_array, np.ndarray):
        transformed = Image.fromarray(transformed_array)
    else:
        transformed = transformed_array
    
    return (original, transformed)


def update_frame_display(frame_idx, frames, fps, annotations, global_annotation, annotation_mode, transformation, process_image_func):
    """Updates frame display"""
    if not frames or frame_idx >= len(frames):
        return {"background": None, "layers": []}, None, f"Frame {int(frame_idx)+1} / 0", "--:--"
    
    # Calculate video time
    if fps > 0:
        current_time = frame_idx / fps
        minutes = int(current_time // 60)
        seconds = current_time % 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"
    else:
        time_str = "--:--"
    
    # Load frame
    frame = frames[int(frame_idx)]
    
    # Create Sketchpad value with transformation
    sketch_value = create_sketchpad_value(frame, annotations, annotation_mode, int(frame_idx), global_annotation, transformation, process_image_func)
    
    # Create comparison slider
    slider_value = create_comparison_slider(frame, transformation, process_image_func)
    
    return sketch_value, slider_value, f"Frame {int(frame_idx)+1} / {len(frames)}", time_str


def go_to_prev_frame(current_idx, steps, frames, fps, annotations, global_annotation, annotation_mode, transformation, process_image_func):
    """Goes one frame back"""
    if not frames:
        return 0, {"background": None, "layers": []}, None, "No video loaded", "--:--"
    
    new_idx = max(0, int(current_idx) - steps)
    sketch_value, slider_value, info, time_str = update_frame_display(new_idx, frames, fps, annotations, global_annotation, annotation_mode, transformation, process_image_func)
    return new_idx, sketch_value, slider_value, info, time_str


def go_to_next_frame(current_idx, steps, frames, fps, annotations, global_annotation, annotation_mode, transformation, process_image_func):
    """Goes one frame forward"""
    if not frames:
        return 0, {"background": None, "layers": []}, None, "No video loaded", "--:--"
    
    new_idx = min(len(frames) - 1, int(current_idx) + steps)
    sketch_value, slider_value, info, time_str = update_frame_display(new_idx, frames, fps, annotations, global_annotation, annotation_mode, transformation, process_image_func)
    return new_idx, sketch_value, slider_value, info, time_str


def load_video_frames(video_path):
    """Loads all frames from a video"""
    if video_path is None:
        return [], 0, gr.update(maximum=0, value=0), "No video loaded", 0, 0, {}, None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    if len(frames) == 0:
        return [], 0, gr.update(maximum=0, value=0), "No frames found", 0, 0, {}, None
    
    duration = len(frames) / fps if fps > 0 else 0
    
    return (
        frames, 
        0, 
        gr.update(maximum=len(frames)-1, value=0),
        f"Frame 1 / {len(frames)}",
        duration,
        fps,
        {},
        None
    )


def save_sketch_annotation(sketch_data, mode, current_frame_idx, annotations, global_annotation):
    """Saves drawing from Sketchpad"""
    annotation_img = extract_annotation_from_sketch(sketch_data)
    
    if annotation_img is None:
        return annotations, global_annotation
    
    new_annotations = annotations.copy() if annotations else {}
    new_global = global_annotation
    
    if mode == "A":
        new_annotations[current_frame_idx] = annotation_img
    else:  # Mode B
        new_global = annotation_img
    
    return new_annotations, new_global


def clear_annotations(mode, annotations, global_annotation):
    """Deletes annotations depending on mode"""
    if mode == "A":
        return {}, global_annotation
    else:  # Mode B
        return annotations, None


def create_tab_videoframes(tab_label, process_image):
    """Creates a tab for video frame processing"""
    with gr.TabItem(tab_label):
        video_frames = gr.State([])
        current_frame_idx = gr.State(0)
        video_duration = gr.State(0)
        video_fps = gr.State(0)
        frame_annotations = gr.State({})
        global_annotation = gr.State(None)
        annotation_mode = gr.State("A")
        selected_transformation = gr.State("None")
        
        
        # Row 1: raw video
        with gr.Accordion("Video Input", open=True):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload video", height=600, sources=['upload'], scale=1)
                
        
        with gr.Row():
            gr.Markdown("---")
            
        # Row 2: video annotations        
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.TabItem("Comparison"):
                        comparison_slider = gr.ImageSlider(
                            label="Original vs Transformed",
                            height=600
                        )
                    with gr.TabItem("Annotations"):
                        with gr.Row():
                            radio_mode = gr.Radio(
                                choices=[("Per Frame", "A"), ("Global", "B")],
                                value="A",
                                label="Annotation Mode",
                                info="Per Frame: Drawings for each frame separately | Global: One drawing over all frames",
                                scale=3
                            )
                            btn_clear_annotations = gr.Button("Clear Annotations", variant="stop", scale=1, size="sm")    
                        
                        with gr.Row():
                            sketch_output = gr.Sketchpad(
                                label="Video Frame (drawing enabled)",
                                height=600,
                                brush=gr.Brush(
                                    colors=["#FF0000", "#00FF00", "#7a7990", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFFFFF", "#000000"],
                                    default_size=3
                                ),
                                type="numpy",
                                scale=2
                            )
            with gr.Column(scale=1, min_width=1):
                

                frame_info = gr.Textbox(label="Frame Info", value="No video loaded", interactive=False, scale=2,)
                video_time_display = gr.Textbox(label="Video Time", value="--:--", interactive=False, scale=1)   
                gr.Markdown("---")
                radio_transformation = gr.Radio(
                    choices=[
                        "None",
                        "Laplacian High-Pass",
                        "FFT Spectrum",
                        "Error Level Analysis",
                        "Wavelet Decomposition",
                        "Noise Extraction",
                        "YCbCr Channels",
                        "Gradient Magnitude",
                        "Histogram Stretching"
                    ],
                    value="None",
                    label="Frame Transformation",
                    info="Apply analysis filters to the current frame"
                )
 
       # with gr.Row():
       #     with gr.Column():
       #         frame_info = gr.Textbox(label="Frame Info", value="No video loaded", interactive=False, scale=2)
       #         video_time_display = gr.Textbox(label="Video Time", value="--:--", interactive=False, scale=1)    
            
        # Row: Frame navigation    
        with gr.Row():
            
            gr.Markdown("---")

        with gr.Row():            
            btn_prev10_frame = gr.Button("◀◀ -10", scale=0, min_width=70)
            btn_prev_frame = gr.Button("◀ -1", scale=0, min_width=70)
            frame_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=0,
                label="Frame Navigation",
                interactive=True,
                scale=20
            )
            btn_next_frame = gr.Button("▶ +1", scale=0, min_width=70)
            btn_next10_frame = gr.Button("▶▶ +10", scale=0, min_width=70)
            
        with gr.Row():
            gr.Markdown("---")
           
            
        # Video Upload
        video_input.change(
            fn=load_video_frames,
            inputs=[video_input],
            outputs=[video_frames, current_frame_idx, frame_slider, frame_info, video_duration, video_fps, frame_annotations, global_annotation]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[current_frame_idx, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        # Frame Navigation
        frame_slider.release(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_prev_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: go_to_prev_frame(idx, 1, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_next_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: go_to_next_frame(idx, 1, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_prev10_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: go_to_prev_frame(idx, 10, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_next10_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: go_to_next_frame(idx, 10, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        # Sketchpad Change - Saves drawing
        sketch_output.change(
            fn=save_sketch_annotation,
            inputs=[sketch_output, annotation_mode, frame_slider, frame_annotations, global_annotation],
            outputs=[frame_annotations, global_annotation]
        )
        
        # Transformation Change
        radio_transformation.change(
            fn=lambda new_trans: new_trans,
            inputs=[radio_transformation],
            outputs=[selected_transformation]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        # Mode Change
        radio_mode.change(
            fn=lambda new_mode: new_mode,
            inputs=[radio_mode],
            outputs=[annotation_mode]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        # Clear Annotations
        btn_clear_annotations.click(
            fn=clear_annotations,
            inputs=[annotation_mode, frame_annotations, global_annotation],
            outputs=[frame_annotations, global_annotation]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )