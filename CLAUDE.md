# YOLO iOS App - Development Guidelines

## Build, Test & Lint Commands
- **Build**: `swift build` or Xcode ⌘+B
- **Run All Tests**: `swift test` or Xcode ⌘+U 
- **Run Single Test**: `swift test --filter YOLOTests/testName`
- **Format**: Automated via GitHub Actions workflow

## Code Style & Conventions
- **Indentation**: 2 spaces
- **Imports**: Alphabetical order (Foundation/UIKit first, custom modules last)
- **Type Safety**: Strong typing with proper protocol usage
- **Naming**: 
  - CamelCase for variables/functions
  - PascalCase for types/protocols
  - Descriptive names with semantic meaning

## Error Handling
- Use Result type for async operations
- Proper completion handlers with Result enum
- Custom error types for specific domains
- Consistent propagation patterns

## Core Technologies
- Swift 6.0 minimum
- iOS 16.0+ deployment target
- CoreML for model inference
- Vision framework for image processing
- Camera and AVFoundation for video capture

## Project Structure
- Swift Package for core YOLO library
- Examples for UIKit and SwiftUI implementations
- Tests separate from implementation code