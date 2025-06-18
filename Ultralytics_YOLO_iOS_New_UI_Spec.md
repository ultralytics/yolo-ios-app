# Ultralytics YOLO iOS App — New UI Specification (June 2025)

> **Purpose**  
> Replace the current UI (see `appstore_preview.png`) with the new design (`Ultralytics_YOLO_design.jpg`).  
> This document is *implementation‑ready*: copy it straight into a ticket or README.

---

## 1. Screen Layout (Hierarchy & Metrics)

| Area | Height | Key Components | Notes |
|------|--------|----------------|-------|
| **Status & Metric Bar** | *Safe‑Area* + 36 pt | Ultralytics icon · Model name drop‑down · Model size · FPS · Latency | Replaces the old task bar |
| **Camera Preview** | Flexible (keeps 16:9) | Live feed · Inference overlay · Right‑hand tool bar | 6 pt black bars on left/right |
| **Task Tab Strip** | 28 pt | DETECT / SEGMENT / CLASS | Anchored to bottom of preview |
| **Shutter Bar** | 96 pt | Thumbnail · Shutter · Camera flip | Center button: 68 pt circle |

---

## 2. Status & Metric Bar

### 2.1 Content Order

```
| UltralyticsLogo | YOLO11 ▼ | SMALL | 27.5 FPS | 26.2 ms |
```

* **Margins:** 12 pt  
* **Font:** SF Pro Rounded Bold 10 pt — white  
* **Drop‑down** opens Model Selector (see 2.2).

### 2.2 Model Selector (Action Sheet)

* Anchored to **Model Name** label.  
* Sections  
  1. **Legacy Models** (YOLOv5x etc.)  
  2. **YOLO‑8 Family**  
  3. **User Custom Models** (includes placeholder)  
* Cell height 36 pt; instant reload on tap.

---

## 3. Camera Preview

### 3.1 Preview Frame

* **Aspect:** 16 : 9 (center‑fit).  
* **CornerRadius:** 18 pt.  
* 6 pt black side‑padding.

### 3.2 Inference Overlay

* Default box colour **Lime `#CFFF1A`**.  
* Label pill: black background, lime stroke, white 8 pt bold text.  
* Redraw at 60 fps.

### 3.3 Right‑side Tool Bar

| # | Icon | Action | Active State |
|---|------|--------|--------------|
| 1 | “1.0x / 1.8x” text | Toggle digital zoom | Lime text |
| 2 | Stacked squares | **Items Max** | Lime fill |
| 3 | Scatter dots | **Confidence Threshold** | Lime fill |
| 4 | ∩ icon | **IoU Threshold** | Lime fill |
| 5 | Pen | **Line Thickness** | Lime fill |

* Button: 40 pt circle, base colour **#6A5545**.  
* Icon stroke white; active = lime fill / black icon.

---

## 4. Parameter Edit Mode

1. **Top Toast** — 120 × 28 pt, radius 14 pt, `#6A5545` 95 % opacity, white text.  
   *Examples:* `ITEMS MAX: 15`, `CONFIDENCE THRESHOLD: 0.82`
2. **Bottom Slider** — 44 pt high, between Preview & ShutterBar.  
   * Track: black.  
   * Ticks: white 1 pt.  
   * Handle: lime stick 4 × 20 pt.  

| Param | Range |
|-------|-------|
| Items Max | 1 – 30 |
| Confidence / IoU | 0.00 – 1.00 (0.02 steps) |
| Line Thickness | 0.5 – 3.0 (0.1) |

Changes apply in real time.

---

## 5. Task Tab Strip

* Tabs: equal width.  
* Font: SF Pro Rounded Semibold 11 pt.  
* Unselected text: `#7D7D7D`.  
* Selected text + underline: **Lime**.  
* 150 ms ease‑out underline animation.

---

## 6. Shutter Bar

| Position | Component | Size |
|----------|-----------|------|
| Left | Last Thumbnail | 48 × 48 pt, radius 8 pt |
| Centre | ShutterBtn | 68 pt circle, white fill, 4 pt black ring |
| Right | Camera Flip | 44 pt circle, white icon |

* Tap → photo (flash overlay 80 ms).  
* Long‑press 0.7 s → video start.

---

## 7. Hidden Info Page

* **Gesture:** Long‑press Ultralytics logo 1 s.  
* Modal page: white bg, logo, caption _“In order to test your models you need to use Hub App”_ plus tiny disclaimer.

---

## 8. Colour & Typography Tokens

| Token | Hex | Usage |
|-------|-----|-------|
| **Primary** | `#CFFF1A` | Accent / selection |
| **Surface‑Dark** | `#000000` | Backgrounds, sliders |
| **Surface‑Brown** | `#6A5545` | Inactive buttons, toast |
| **Text‑Primary** | `#FFFFFF` | Main text |
| **Text‑Subtle** | `#7D7D7D` | Inactive tabs |

Font everywhere: **SF Pro Rounded** (fallback SF Pro).

---

## 9. Animation & Gestures

| Trigger | Effect | Duration |
|---------|--------|----------|
| Model name tap | Action Sheet down | 250 ms |
| Side button tap | Toast + Slider fade in | 150 ms |
| Slider idle 0.5 s | Toast fade out | 300 ms |
| Photo capture | Full‑screen flash | 80 ms |

---

## 10. Implementation Notes

* **AutoLayout Priority**  
  * Preview: fixed 16 : 9 ratio, <= (super.height – TabStrip – ShutterBar).  
  * Slider slides in without resizing Preview.  
* **Persistence**  
  * `ItemsMax`, `Confidence`, `IoU`, `Thickness` stored in `UserDefaults`.  
* **Isolation**  
  * Re‑implement only the **UI layer**; existing CV pipeline remains untouched.

---

*End of file*  
