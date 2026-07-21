---
title: Agentic Backend
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Agentic Backend

This application is associated with Mumbai Housing Society Redevelopment.
THe backend expose endpoints which will be consumed by the Frontend build on top Streamlit and hosted in Streamlit cloud services.
This services work with FastAPI, Langchain ecosystem and uses Agentic AI
FastAPI backend for the Agentic project, deployed on Hugging Face Spaces.

## Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```