set -euo pipefail
 
BACKEND_DIR="/home/ubuntu/projects/Agentic/backend"
GITHUB_REPO_NAME="agentic_backend"
VISIBILITY="public"
USERNAME="revglen"
 
cd "$BACKEND_DIR"

git config --global user.email "revglen@gmail.com"
git config --global user.name "revglen"
 
git init
git branch -M main
git add .
git commit -m "Initial backend commit"
 
#gh repo create "$GITHUB_REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
git remote add origin https://github.com/$USERNAME/$GITHUB_REPO_NAME.git
git push -u origin main