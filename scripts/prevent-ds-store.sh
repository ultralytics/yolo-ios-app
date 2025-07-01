#!/bin/bash
# Script to prevent creation of .DS_Store files on network volumes
# and help manage them in the repository

echo "Setting up .DS_Store prevention..."

# Prevent .DS_Store files on network volumes
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true

# Add .DS_Store to global gitignore if not already present
GLOBAL_GITIGNORE="$HOME/.gitignore_global"
if [ ! -f "$GLOBAL_GITIGNORE" ]; then
    echo ".DS_Store" > "$GLOBAL_GITIGNORE"
    echo "Created global gitignore with .DS_Store"
else
    if ! grep -q "^\.DS_Store$" "$GLOBAL_GITIGNORE"; then
        echo ".DS_Store" >> "$GLOBAL_GITIGNORE"
        echo "Added .DS_Store to global gitignore"
    else
        echo ".DS_Store already in global gitignore"
    fi
fi

# Set global gitignore
git config --global core.excludesfile "$GLOBAL_GITIGNORE"

echo "Setup complete!"
echo ""
echo "To remove existing .DS_Store files from the repository, run:"
echo "find . -name '.DS_Store' -type f -delete"
echo ""
echo "To remove them from git history (if needed):"
echo "git rm --cached -r . -f && git add . && git commit -m 'Remove .DS_Store files'"