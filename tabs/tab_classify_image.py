#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 09:52:28 2026

@author: standarduser
"""

from gradio_client import Client, handle_file
import gradio as gr

def predict_from_space(image_path):
    """Classify image using Space API."""
    
    client = Client("ElBeh/image-fake-detector")
    
    try:
        result = client.predict(
            image=handle_file(image_path),
            api_name="/predict"
        )
        
        # Gradio Label format: {'label': 'Fake', 'confidences': [{'label': 'Fake', 'confidence': 0.88}, ...]}
        confidences = result['confidences']
        
        # Extract probabilities from confidences list
        proba_dict = {item['label']: item['confidence'] for item in confidences}
        proba_real = proba_dict.get('Real', 0.0)
        proba_fake = proba_dict.get('Fake', 0.0)
        
        # Determine prediction
        prediction = 1 if proba_fake > 0.5 else 0
        label = "Fake" if prediction == 1 else "Real"
        confidence = proba_fake if prediction == 1 else proba_real
        
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"Real: {proba_real:.4f} | Fake: {proba_fake:.4f}")
        
        return {
            'Real': float(proba_real),
            'Fake': float(proba_fake)
        }
    
    except Exception as e:
        print(f"Error: {e}")
        raise

def create_tab_classify_image(tab_label):
    with gr.TabItem(tab_label):
            gr.Interface(
                fn=predict_from_space,
                inputs=[
                gr.Image(type="filepath", label="Upload Image"),  
            ],
            outputs=gr.Label(num_top_classes=2, label="Prediction"),
            title="Image Fake Detector",
            description="Upload an image to classify it as real or fake. The detector(XGBoost) uses several image statistics to classify the image."
        )