# Deploying Backend to Hugging Face Spaces

This guide explains how to deploy the `backend` folder to a Hugging Face Space.

## Routine Workflow

For your daily development, follow these steps:

1.  **Commit Changes Locally** (Same as always)
    ```bash
    git add .
    git commit -m "Your commit message"
    ```

2.  **Push Full Project to GitHub** (Backup & Version Control)
    ```bash
    git push origin main
    ```

3.  **Deploy Backend to Hugging Face** (Update API)
    Run this **from the root** of your project:
    ```bash
    git subtree push --prefix backend hf main
    ```

---

## Setup & Troubleshooting

### Prerequisites

1.  Create a **new Space** on Hugging Face (SDK: Docker).
2.  Add Remote (if not already added):
    ```bash
    git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
    ```

### Troubleshooting
*   **Authentication**: If prompted for a password, use your **Hugging Face Access Token** (with write permissions).
*   **Conflict / Rejected Push**: If `git subtree push` fails because of remote changes (e.g., you edited files on the HF website), use a force push:
    ```bash
    git push hf `git subtree split --prefix backend main`:main --force
    ```
