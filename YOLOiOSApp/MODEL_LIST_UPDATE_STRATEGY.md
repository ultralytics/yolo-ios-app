# Model List System Update Strategy

## Overview
Update the model list system to support YOLOv8 and YOLOv5 models alongside YOLO11, with a new size-based filtering UI.

## Current State Analysis

### Current Model Structure
- Models are identified by `ModelEntry` with properties:
  - `displayName`: "yolo11n", "yolo11s", etc.
  - `identifier`: Same as displayName for local, filename for remote
  - `isLocalBundle`: Boolean for bundled models
  - `isRemote`: Boolean for downloadable models
  - `remoteURL`: URL for remote models

### Current UI Flow
1. Status bar shows current model name and size
2. Model dropdown shows all models for current task
3. Models are grouped by: Selected, Downloaded, Available

## Proposed Changes

### 1. Model Naming Convention
Update model naming to include version explicitly:
- YOLO11: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- YOLOv5: `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`

### 2. Size Filter UI Component
Create a new `ModelSizeFilterBar` component:
- Horizontal stack of size buttons: NANO, SMALL, MEDIUM, LARGE, XLARGE
- Appears below StatusMetricBar when size button is tapped
- Updates the current size filter
- Hides when a size is selected or tapped outside

### 3. Model Filtering Logic
Implement multi-level filtering:
1. **Task Filter**: Only show models for current task (existing)
2. **Size Filter**: Only show models matching selected size
3. **Version Groups**: Group models by version in dropdown

### 4. Model Display Structure
New dropdown organization:
```
[Selected Model]
────────────────
YOLO11
YOLOv8  
YOLOv5
────────────────
CUSTOM MODELS
Model A
Model B
```

### 5. Data Structure Updates

#### Update RemoteModels.swift
Add entries for YOLOv8 and YOLOv5 with placeholder URLs:
```swift
// Detection Models
"yolov8n": "https://placeholder.com/yolov8n.mlpackage.zip",
"yolov8s": "https://placeholder.com/yolov8s.mlpackage.zip",
// ... etc for all sizes and tasks
```

#### Update ModelEntry
Add computed properties:
```swift
var modelVersion: String // "YOLO11", "YOLOv8", "YOLOv5", "Custom"
var modelSize: String? // "n", "s", "m", "l", "x", nil for custom
```

### 6. Implementation Steps

#### Phase 1: Data Layer
1. Update `RemoteModels.swift` with YOLOv8/v5 URLs
2. Add helper methods to `ModelEntry` for version/size extraction
3. Update `ModelSizeHelper` to handle new naming conventions

#### Phase 2: Size Filter UI
1. Create `ModelSizeFilterBar.swift` component
2. Add show/hide animations
3. Connect to StatusMetricBar size button tap

#### Phase 3: Model Dropdown Updates
1. Update `ModelDropdownView` to support version grouping
2. Implement size-based filtering
3. Update cell design to show only model version

#### Phase 4: Integration
1. Update `ViewController` to manage size filter state
2. Connect size filter to model dropdown
3. Update model selection logic

### 7. UI Flow

1. **Initial State**: 
   - Status bar shows "YOLO11 SMALL"
   - Size filter bar is hidden

2. **Size Button Tap**:
   - Size filter bar slides down
   - Shows 5 size options
   - Current size is highlighted

3. **Size Selection**:
   - Updates current size filter
   - Refreshes model dropdown
   - Hides size filter bar

4. **Model Dropdown**:
   - Shows only models matching current size
   - Groups by version
   - Custom models always visible

### 8. Edge Cases
- Custom models: No size filtering
- Missing models: Handle gracefully
- Download states: Maintain existing logic
- Orientation changes: Adjust layout

### 9. Testing Strategy
1. Test all size/version combinations
2. Verify custom model visibility
3. Test download/cache functionality
4. Verify UI transitions