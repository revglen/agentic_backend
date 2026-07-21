#!/usr/bin/env bash
# Safe to run more than once.
# Prerequisite (one-time, if not already done): gh auth login && gh auth setup-git
 
set -euo pipefail

if [ -f ../.env ]; then
  export $(cat ../.env | grep -v '^#' | xargs)
  echo "The Environment variables have been set"
fi
 
BACKEND_DIR="/home/ubuntu/projects/Agentic/backend"
GITHUB_REPO_NAME="agentic_backend"
USERNAME="revglen"
#REMOTE_URL="https://github.com/$USERNAME/$GITHUB_REPO_NAME.git"
REMOTE_URL="https://$USERNAME:$GITHUB_TOKEN@github.com/$USERNAME/$GITHUB_REPO_NAME.git"
 
cd "$BACKEND_DIR"
 
git config --global user.email "revglen@gmail.com"
git config --global user.name "revglen"
 
if [ ! -d .git ]; then
  git init
  git branch -M main
fi
 
git add .
git commit -m "Update backend" || echo "Nothing new to commit"
 
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi
 
git push -u origin main
 