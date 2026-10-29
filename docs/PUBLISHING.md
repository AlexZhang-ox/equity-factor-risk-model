# Manual publishing guide

These instructions are for the repository owner to carry out. Preparing this folder did not create a GitHub repository, configure a remote, or upload any files.

Suggested repository name: `equity-factor-risk-model`

Suggested description: `Cross-sectional equity factor estimation, structural risk modeling, and a transparent long-short portfolio case study in Python.`

Choose Public or Private according to your preference. No license has been selected. Add a license only after deliberately choosing how to license material you are entitled to publish; the private mentor note and raw vendor prices are not part of this folder.

## 1. Open the publication folder

In VS Code, use **File > Open Folder** and choose the clean `equity-factor-risk-model` folder supplied for publication. Do not initialize or upload the older working project containing its environment and cached market data. A separate window can make this easier if another workspace has several folders open.

The public inventory is:

```text
README.md
equity_lfm_project.py
equity_lfm_project.ipynb
requirements.txt
requirements-notebook.txt
.gitignore
.gitattributes
.github/workflows/tests.yml
docs/METHODOLOGY.md
docs/PUBLISHING.md
data/README.md
reports/performance_summary.csv
reports/run_config.json
reports/validation_summary.json
reports/figures/performance.png
reports/figures/factor_diagnostics.png
reports/figures/latest_weights.png
```

Keep the negative recorded results and research limitations in the README. The notebook is intentionally published without saved outputs, and its explanations are in English.

Before uploading, inspect the files you will publish. Exclude `.venv`, local editor settings, raw cached prices, runtime `results` directories, private PDFs, credentials, and files containing local absolute paths. The supplied ignore rules handle these for normal Git staging, but do not replace inspection.

## 2. Publish with VS Code

1. Open **Source Control** and select **Initialize Repository** for the publication folder.
2. Review the changed files. The visible list should match the public inventory, without environments or downloaded prices.
3. Stage the intended files and inspect the staged changes.
4. Commit with an English message such as `Add equity factor risk model case study`.
5. Select **Publish Branch**, sign in to your GitHub account if needed, choose the name `equity-factor-risk-model`, and choose Public or Private.
6. Open the resulting repository on GitHub and inspect the rendered README, relative image links, and notebook.

For this route, let VS Code create the remote repository. If you already created an empty repository on the GitHub website, use the terminal route below to connect the local repository to that existing remote.

## 3. Alternative: publish with Git in a terminal

On GitHub, create a new repository named `equity-factor-risk-model`. Choose visibility yourself. When pushing an existing local project, create an empty remote: do not ask GitHub to add a README, `.gitignore`, or license during creation.

Open a terminal in the clean publication folder and run these commands yourself:

```bash
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git diff --cached
```

Stop to inspect the staged files and content. If anything private or generated is present, unstage it and correct the file selection before continuing. Do not force-add ignored files to make the inventory larger.

Then commit:

```bash
git commit -m "Add equity factor risk model case study"
```

Replace `YOUR_ACCOUNT` in the next command with your own GitHub account or organization, or use the exact remote URL shown for your new repository. Do not paste the placeholder unchanged.

```bash
git remote add origin "https://github.com/YOUR_ACCOUNT/equity-factor-risk-model.git"
git push -u origin main
```

Authenticate through your normal GitHub sign-in flow when requested. These commands are provided as instructions; they have not been run against a remote as part of preparing this project.

## 4. Alternative: upload in the browser

Create the GitHub repository yourself, then use **Add file > Upload files** or the initial page's existing-file upload option. Upload the curated public contents while preserving their directory structure. Do not upload a ZIP as a substitute for the source tree.

Important: GitHub's browser upload does not apply your local `.gitignore`. If you have run this project, the folder may now contain an environment, cached prices, and generated results. Select only the public inventory above; do not blindly drag the entire working folder.

On macOS, **Command+Shift+.** reveals hidden files in a file picker. Include `.gitignore`, `.gitattributes`, and the `.github` workflow directory. Verify their presence and nested file paths in GitHub before committing the upload.

Use an English commit message such as `Add equity factor risk model case study`.

## 5. Check the published repository

- The README figures should render from `reports/figures/`.
- Core source, notebook text, documentation, and saved reports should be in English.
- Raw vendor price files, the mentor PDF, environments, secrets, and local machine paths should be absent.
- A fresh clone should be able to install requirements and run `python equity_lfm_project.py --self-test` without downloading market data.
- A full research run requires Yahoo access and may produce different values after vendor history revisions.
- The supplied CI workflow can run after publication. A local validation report is not a claim that GitHub CI has already passed.

No GitHub repository has been created or uploaded during preparation of this folder. Publication remains your manual action.
