#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Build Flutter app for iOS with specified flavor and target
flutter build ios --release --flavor production --target lib/main_production.dart

# Create Payload folder and copy Runner.app inside it
mkdir Payload
cp -r build/ios/iphoneos/Runner.app Payload/

# Zip the Payload folder
zip -r Payload.zip Payload

# Rename the zip file to latest.ipa
mv Payload.zip latest.ipa

# Clean up: remove the Payload folder and original Runner.app
rm -rf Payload build

echo "Build completed successfully. The IPA file is named latest.ipa."
