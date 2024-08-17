#!/usr/bin/env bash

# Update package lists
apt-get update 

# Install Tesseract OCR
apt-get install -y tesseract-ocr

# (Optional) Install additional language data if needed
# apt-get install -y tesseract-ocr-all
