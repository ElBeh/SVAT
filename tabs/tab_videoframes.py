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
import tempfile
import os

# Import classification function
from tabs.tab_classify_image import predict_from_space

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

def apply_transformation(frame, transformation, quality, process_image_func):
    """Applies selected transformation to frame"""
    if frame is None or transformation == "None":
        return frame
    
    # Convert numpy array to PIL if needed
    if isinstance(frame, np.ndarray):
        pil_frame = Image.fromarray(frame)
    else:
        pil_frame = frame
    
    # Call process_image with the frame and quality
    result = process_image_func(pil_frame, transformation, quality)
    
    # Extract transformed image from tuple
    if isinstance(result, tuple) and len(result) == 2:
        transformed = result[1]
    else:
        transformed = result
    
    # CRITICAL FOR GRADIO 6.x: Convert grayscale to RGB
    if transformed is not None:
        if isinstance(transformed, Image.Image) and transformed.mode == 'L':
            transformed = transformed.convert('RGB')
        elif isinstance(transformed, np.ndarray) and len(transformed.shape) == 2:
            transformed = Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_GRAY2RGB))
        
        # Convert to numpy array
        return np.array(transformed)
    
    return frame

def create_sketchpad_value(base_image, annotations, mode, current_frame_idx, global_annotation, transformation, quality, process_image_func):
    """Creates Sketchpad value (Background + Layers)"""
    if base_image is None:
        return None
    
    # Apply transformation first
    transformed_frame = apply_transformation(base_image, transformation, quality, process_image_func)
    
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

def create_comparison_slider(frame, transformation, quality, process_image_func):
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
    transformed_array = apply_transformation(frame, transformation, quality, process_image_func)
    
    if isinstance(transformed_array, np.ndarray):
        transformed = Image.fromarray(transformed_array)
    else:
        transformed = transformed_array
    
    return (original, transformed)


# NEW: Classification functions
def classify_current_frame(frame_idx, frames, existing_classifications):
    """Classify current frame and cache result"""
    frame_idx = int(frame_idx)
    
    # Check if already classified
    if frame_idx in existing_classifications:
        return (
            existing_classifications[frame_idx],
            f"✓ Cached result (Frame {frame_idx + 1})",
            existing_classifications
        )
    
    if not frames or frame_idx >= len(frames):
        return None, "✗ No frame available", existing_classifications
    
    frame = frames[frame_idx]
    
    # Save temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        Image.fromarray(frame).save(tmp.name, 'JPEG', quality=95)
        tmp_path = tmp.name
    
    try:
        result = predict_from_space(tmp_path)
        
        # Cache result
        new_classifications = existing_classifications.copy()
        new_classifications[frame_idx] = result
        
        return result, f"✓ Frame {frame_idx + 1} classified", new_classifications
    
    except Exception as e:
        return None, f"✗ API Error: {str(e)}", existing_classifications
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def update_classification_display(frame_idx, classifications):
    """Update classification display when switching frames"""
    frame_idx = int(frame_idx)
    
    if frame_idx in classifications:
        return classifications[frame_idx], f"✓ Frame {frame_idx + 1} (cached)"
    else:
        return None, "Not classified yet"


def update_frame_display(frame_idx, frames, fps, annotations, global_annotation, annotation_mode, transformation, quality, process_image_func):
    """Updates frame display"""
    if not frames or frame_idx >= len(frames):
        return (
            {"background": None, "layers": [], "composite": None},  # Fixed: Added 'composite' key
            None, 
            f"Frame {int(frame_idx)+1} / 0", 
            "--:--"
        )
    
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
    sketch_value = create_sketchpad_value(frame, annotations, annotation_mode, int(frame_idx), global_annotation, transformation, quality, process_image_func)
    
    # Create comparison slider
    slider_value = create_comparison_slider(frame, transformation, quality, process_image_func)
    
    return sketch_value, slider_value, f"Frame {int(frame_idx)+1} / {len(frames)}", time_str


def go_to_prev_frame(current_idx, steps, frames, fps, annotations, global_annotation, annotation_mode, transformation, quality, process_image_func):
    """Goes one frame back"""
    if not frames:
        return 0, {"background": None, "layers": []}, None, "No video loaded", "--:--"
    
    new_idx = max(0, int(current_idx) - steps)
    sketch_value, slider_value, info, time_str = update_frame_display(new_idx, frames, fps, annotations, global_annotation, annotation_mode, transformation, quality, process_image_func)
    return new_idx, sketch_value, slider_value, info, time_str


def go_to_next_frame(current_idx, steps, frames, fps, annotations, global_annotation, annotation_mode, transformation, quality, process_image_func):
    """Goes one frame forward"""
    if not frames:
        return 0, {"background": None, "layers": []}, None, "No video loaded", "--:--"
    
    new_idx = min(len(frames) - 1, int(current_idx) + steps)
    sketch_value, slider_value, info, time_str = update_frame_display(new_idx, frames, fps, annotations, global_annotation, annotation_mode, transformation, quality, process_image_func)
    return new_idx, sketch_value, slider_value, info, time_str


def load_video_frames(video_path):
    """Loads all frames from a video"""
    if video_path is None:
        return [], 0, gr.update(maximum=0, value=0), "No video loaded", 0, 0, {}, None, {}  # Added {} for frame_classifications
    
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
        return [], 0, gr.update(maximum=0, value=0), "No frames found", 0, 0, {}, None, {}  # Added {} for frame_classifications
    
    duration = len(frames) / fps if fps > 0 else 0
    
    return (
        frames, 
        0, 
        gr.update(maximum=len(frames)-1, value=0),
        f"Frame 1 / {len(frames)}",
        duration,
        fps,
        {},
        None,
        {}  # Reset frame_classifications
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


def toggle_accordion(accordion_name, current_active):
    """Toggles accordion visibility and returns new transformation state with button variants"""
    transformation_names = [
        "Laplacian High-Pass",
        "FFT Spectrum",
        "Error Level Analysis",
        "Wavelet Decomposition",
        "Noise Extraction",
        "YCbCr Channels",
        "Gradient Magnitude",
        "Histogram Stretching"
    ]
    
    if current_active == accordion_name:
        # Clicking active accordion closes it -> None
        new_transformation = "None"
        visibility = [False] * 8
        variants = ["secondary"] * 8  # All buttons secondary (gray)
    else:
        # Open clicked accordion, close all others
        new_transformation = accordion_name
        visibility = [accordion_name == name for name in transformation_names]
        # Set clicked button to primary (highlighted), others to secondary
        variants = ["primary" if accordion_name == name else "secondary" for name in transformation_names]
    
    return (new_transformation, 
            *[gr.update(visible=v) for v in visibility],
            *[gr.update(variant=var) for var in variants])


def create_tab_videoframes(tab_label, process_image, shared_video_frames=None):
    """Creates a tab for video frame processing"""
    with gr.TabItem(tab_label):
        # Use shared state if provided, otherwise create local state
        if shared_video_frames is None:
            video_frames = gr.State([])
        else:
            video_frames = shared_video_frames
            
        current_frame_idx = gr.State(0)
        video_duration = gr.State(0)
        video_fps = gr.State(0)
        frame_annotations = gr.State({})
        global_annotation = gr.State(None)
        annotation_mode = gr.State("A")
        selected_transformation = gr.State("None")
        ela_quality = gr.State(90)
        frame_classifications = gr.State({})  # NEW: Store classification results
        
        
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
                frame_info = gr.Textbox(label="Frame Info", value="No video loaded", interactive=False, scale=2)
                video_time_display = gr.Textbox(label="Video Time", value="--:--", interactive=False, scale=1)   
                
                gr.Markdown("---")
                
                # Accordion-based transformation selection
                with gr.Column():
                    gr.Markdown("### Frame Transformation")
                    gr.Markdown("*Click to activate transformation*")
                    
                    # Laplacian High-Pass
                    btn_laplacian = gr.Button("▶ Laplacian High-Pass", size="sm")
                    with gr.Column(visible=False) as content_laplacian:
                        gr.Markdown("Emphasizes high-frequency details and edges")
                    
                    # FFT Spectrum
                    btn_fft = gr.Button("▶ FFT Spectrum", size="sm")
                    with gr.Column(visible=False) as content_fft:
                        gr.Markdown("Shows frequency domain representation")
                    
                    # Error Level Analysis
                    btn_ela = gr.Button("▶ Error Level Analysis", size="sm")
                    with gr.Column(visible=False) as content_ela:
                        gr.Markdown("Detects JPEG compression artifacts")
                        quality_slider = gr.Slider(
                            minimum=1,
                            maximum=99,
                            value=90,
                            step=1,
                            label="JPEG Quality",
                            info="Higher = more subtle differences"
                        )
                    
                    # Wavelet Decomposition
                    btn_wavelet = gr.Button("▶ Wavelet Decomposition", size="sm")
                    with gr.Column(visible=False) as content_wavelet:
                        gr.Markdown("Multi-scale frequency analysis")
                    
                    # Noise Extraction
                    btn_noise = gr.Button("▶ Noise Extraction", size="sm")
                    with gr.Column(visible=False) as content_noise:
                        gr.Markdown("Isolates high-frequency noise")
                    
                    # YCbCr Channels
                    btn_ycbcr = gr.Button("▶ YCbCr Channels", size="sm")
                    with gr.Column(visible=False) as content_ycbcr:
                        gr.Markdown("Separates luminance and chrominance")
                    
                    # Gradient Magnitude
                    btn_gradient = gr.Button("▶ Gradient Magnitude", size="sm")
                    with gr.Column(visible=False) as content_gradient:
                        gr.Markdown("Visualizes edge strength via Sobel")
                    
                    # Histogram Stretching
                    btn_histogram = gr.Button("▶ Histogram Stretching", size="sm")
                    with gr.Column(visible=False) as content_histogram:
                        gr.Markdown("Extreme contrast enhancement")
 
        # Row: Frame Classification
        with gr.Row():
            gr.Markdown("---")
        
        with gr.Accordion("Frame Classification - (optimized model for ai images)", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        btn_classify_frame = gr.Button("🔍 Classify Current Frame", size="sm", variant="primary")
                        btn_classify_all = gr.Button("🔬 Classify All Frames (Coming Soon)", size="sm", interactive=False)
                with gr.Column(scale=2):
                    classification_result = gr.Label(num_top_classes=2, label="Result")
                with gr.Column(scale=1):
                    classification_status = gr.Textbox(label="Status", value="Not classified yet", interactive=False)
        
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
        
        # Collect all content columns for visibility updates
        content_columns = [
            content_laplacian,
            content_fft,
            content_ela,
            content_wavelet,
            content_noise,
            content_ycbcr,
            content_gradient,
            content_histogram
        ]
        
        # Collect all buttons for variant updates
        transformation_buttons = [
            btn_laplacian,
            btn_fft,
            btn_ela,
            btn_wavelet,
            btn_noise,
            btn_ycbcr,
            btn_gradient,
            btn_histogram
        ]
        
        # NEW: Classification button event
        btn_classify_frame.click(
            fn=classify_current_frame,
            inputs=[frame_slider, video_frames, frame_classifications],
            outputs=[classification_result, classification_status, frame_classifications]
        )
        
        # Accordion button clicks
        btn_laplacian.click(
            fn=lambda current: toggle_accordion("Laplacian High-Pass", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_fft.click(
            fn=lambda current: toggle_accordion("FFT Spectrum", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_ela.click(
            fn=lambda current: toggle_accordion("Error Level Analysis", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_wavelet.click(
            fn=lambda current: toggle_accordion("Wavelet Decomposition", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_noise.click(
            fn=lambda current: toggle_accordion("Noise Extraction", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_ycbcr.click(
            fn=lambda current: toggle_accordion("YCbCr Channels", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_gradient.click(
            fn=lambda current: toggle_accordion("Gradient Magnitude", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        btn_histogram.click(
            fn=lambda current: toggle_accordion("Histogram Stretching", current),
            inputs=[selected_transformation],
            outputs=[selected_transformation] + content_columns + transformation_buttons
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        # Quality slider change (only affects ELA)
        quality_slider.change(
            fn=lambda q: q,
            inputs=[quality_slider],
            outputs=[ela_quality]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
           
        # Video Upload - MODIFIED: Added frame_classifications to outputs
        video_input.change(
            fn=load_video_frames,
            inputs=[video_input],
            outputs=[video_frames, current_frame_idx, frame_slider, frame_info, video_duration, video_fps, frame_annotations, global_annotation, frame_classifications]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[current_frame_idx, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        ).then(
            fn=lambda: (None, "Not classified yet"),  # Reset classification display
            inputs=[],
            outputs=[classification_result, classification_status]
        )
        
        # Frame Navigation - MODIFIED: Added classification display update
        frame_slider.release(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        ).then(
            fn=update_classification_display,
            inputs=[frame_slider, frame_classifications],
            outputs=[classification_result, classification_status]
        )
        
        btn_prev_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: go_to_prev_frame(idx, 1, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        ).then(
            fn=update_classification_display,
            inputs=[frame_slider, frame_classifications],
            outputs=[classification_result, classification_status]
        )
        
        btn_next_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: go_to_next_frame(idx, 1, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        ).then(
            fn=update_classification_display,
            inputs=[frame_slider, frame_classifications],
            outputs=[classification_result, classification_status]
        )
        
        btn_prev10_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: go_to_prev_frame(idx, 10, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        ).then(
            fn=update_classification_display,
            inputs=[frame_slider, frame_classifications],
            outputs=[classification_result, classification_status]
        )
        
        btn_next10_frame.click(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: go_to_next_frame(idx, 10, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[frame_slider, sketch_output, comparison_slider, frame_info, video_time_display]
        ).then(
            fn=update_classification_display,
            inputs=[frame_slider, frame_classifications],
            outputs=[classification_result, classification_status]
        )
        
        # Sketchpad Change - Saves drawing
        sketch_output.change(
            fn=save_sketch_annotation,
            inputs=[sketch_output, annotation_mode, frame_slider, frame_annotations, global_annotation],
            outputs=[frame_annotations, global_annotation]
        )
        
        # Mode Change
        radio_mode.change(
            fn=lambda new_mode: new_mode,
            inputs=[radio_mode],
            outputs=[annotation_mode]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )
        
        # Clear Annotations
        btn_clear_annotations.click(
            fn=clear_annotations,
            inputs=[annotation_mode, frame_annotations, global_annotation],
            outputs=[frame_annotations, global_annotation]
        ).then(
            fn=lambda idx, frames, fps, annots, glob_annot, mode, trans, quality: update_frame_display(idx, frames, fps, annots, glob_annot, mode, trans, quality, process_image),
            inputs=[frame_slider, video_frames, video_fps, frame_annotations, global_annotation, annotation_mode, selected_transformation, ela_quality],
            outputs=[sketch_output, comparison_slider, frame_info, video_time_display]
        )

        # Return video_frames state for sharing with other tabs
        return video_frames