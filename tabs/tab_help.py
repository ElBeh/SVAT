#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 11:58:29 2025

@author: standarduser
"""



import gradio as gr
        
        
def create_tab_help(tab_label):
    """create help tab"""
    with gr.TabItem(tab_label):
        gr.Markdown("# Help")

        
