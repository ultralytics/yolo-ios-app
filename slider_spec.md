# YOLO Camera App – Threshold Slider Specification

## 1. Layout & Layer Structure
- **Container**
  - Height: ~48 pt; positioned within safe area (above the Home indicator).
  - Full‑width, with 8 pt horizontal padding to avoid edge‑swipe conflicts.
  - Background: solid black `#0B0F15`.

- **Base track**
  - Drawn with a `CALayer`; 4 pt height; color `#141A23`.

- **Major / minor ticks**
  - **Major:** 10 pt spacing, 14 pt length, 1 pt stroke, pure white.
  - **Minor:** halfway marks, 8 pt length, 50 %‑white.
  - Use a single `CAShapeLayer` path to minimise layer count.

- **Thumb (cursor)**
  - Size 24 × 2 pt, corner radius 1 pt, neon green `#C1FF00`.
  - Hit‑box is expanded ±20 pt horizontally.
  - Position:  
    ```swift
    thumbCenterX = padding + value * effectiveWidth // value ∈ 0…1
    ```
- **Focus glow**
  - While dragging or VoiceOver‑focused, animate `shadowRadius` 5 → 10 via `CABasicAnimation`.

## 2. Value Representation & Binding
- Internal range **0 – 1** (`Float`).
- `ObservableObject` example:
  ```swift
  @Published var confidence: Float = 0.5
  ```
- `DragGesture` updates the model in real time; `didSet` pushes the value to Core ML / TFLite.

## 3. Step Snap & Haptics
- Snap to **0.05** increments  
  ```swift
  value = round(value * 20) / 20
  ```
- On snap completion trigger  
  ```swift
  UIImpactFeedbackGenerator(style: .light).impactOccurred()
  ```

## 4. Animation
- **Appear:** opacity 0 → 1 & translate Y 20 → 0 over 0.25 s.
- **Idle dim:** after 3 s of no interaction, fade `alpha` to 0.3; restore to 1.0 on touch.

## 5. Accessibility
- Set `accessibilityTraits = .adjustable`.
- Implement `accessibilityIncrement/Decrement` (±0.05).
- `accessibilityValue` speaks the percentage (e.g. “50 percent”).

## 6. Color Dynamics
- Light‑mode fallback:
  - Track: `systemGray5`
  - Ticks: `label`
- Thumb may follow `tintColor`, or stay constant neon green for brand identity.

## 7. Performance Optimisation
- Static graphics (track + ticks) rendered once into a `CATiledLayer`.
- Only the thumb layer is animated each frame.
- Maintain Z‑order: **camera preview < slider < toolbar buttons**.
