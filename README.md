# Documentation Site Management Guide

Based on your project structure, I'll provide you with steps to update your site and automate the deployment flow.

## Project Overview

Your documentation site is built with MkDocs using the ReadTheDocs theme. Here's how to manage and deploy updates.

## Steps to Update Site Content

1. **Edit Documentation Files**
   - Modify existing files in the `docs/` directory
   - Add new Markdown files for new pages
   - Update navigation in [mkdocs.yml](file:///home/luyao/workspace/ly-docs/mkdocs.yml) if needed

2. **Preview Changes Locally**
   ```bash
   source /home/luyao/workspace/doc_py/bin/activate
   cd /home/luyao/workspace/ly-docs
   mkdocs serve
   ```
   Visit `http://127.0.0.1:8000` to preview your changes.

3. **Commit Changes**
   ```bash
   cd /home/luyao/workspace/ly-docs
   git add .
   git commit -m "Describe your changes"
   git push origin main  # or master
   ```

4. **Deploy to GitHub Pages**
   ```bash
   source /home/luyao/workspace/doc_py/bin/activate
   cd /home/luyao/workspace/ly-docs
   mkdocs gh-deploy
   ```

## Automating the Deployment Flow

To automate deployment, create a GitHub Actions workflow:

1. Create directory structure:
   ```
   mkdir -p .github/workflows
   ```

2. Create `.github/workflows/deploy.yml` with the following content:

```yaml
name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main  # or master, depending on your default branch

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          pip install mkdocs
      - name: Deploy to GitHub Pages
        run: |
          mkdocs gh-deploy --force
```

With this workflow:
- Each time you push to the main branch, GitHub Actions will automatically build and deploy your site
- Your site at `https://frank2679.github.io/cutlass-doc/` will stay up-to-date
- No manual deployment steps are needed

## Adding New Pages

1. Create a new Markdown file in the `docs/` directory
2. Add content using Markdown syntax
3. Update the navigation in [mkdocs.yml](file:///home/luyao/workspace/ly-docs/mkdocs.yml):
   ```yaml
   nav:
     - Home: index.md
     - New Section:
       - new-page.md
   ```

## Tips for Efficient Workflow

1. Always preview locally before deploying
2. Write clear commit messages
3. Test links and formatting before pushing
4. Keep the navigation structure organized

After setting up the GitHub Actions workflow, your documentation site will automatically update whenever you push changes to your main branch.