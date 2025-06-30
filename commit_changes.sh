#!/bin/bash

# Navigate to the repository
cd /Users/majimadaisuke/Downloads/yolo-ios-app

# Add only the source files we modified (no build files or models)
git add Sources/YOLO/BasePredictor.swift
git add Sources/YOLO/YOLOView.swift
git add YOLOiOSApp/YOLOiOSApp/ViewController.swift
git add YOLOiOSApp/YOLOiOSApp/ModelMetadataHelper.swift
git add YOLOiOSApp/YOLOiOSApp/ModelDownloadManager.swift

# Show what will be committed
echo "Files to be committed:"
git status --porcelain | grep "^M\|^A"

# Create commit
git commit -m "Implement automatic model size detection from metadata

- Add metadata extraction and caching for custom models
- Fix custom model size detection (was incorrectly showing 'large')
- Add ModelMetadataHelper to extract size from model description field
- Update status bar to show correct size (NANO, SMALL, etc.)
- Default to NANO size for custom models without size metadata
- Remove duplicate code and consolidate metadata extraction logic"

# Show commit info
git log -1 --stat

echo "Commit complete. Run 'git push' to push changes."