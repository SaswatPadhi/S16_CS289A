#!/bin/bash

# Remove local and remote tag
git tag -d PDFs
git push origin :PDFs

# Create new tag (replace old one)
git tag -f PDFs

# Push commits & tags
git push && git push --tags
