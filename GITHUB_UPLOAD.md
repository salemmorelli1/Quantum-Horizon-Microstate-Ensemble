# Upload with Git Bash

The repository is configured for the remote name:

`https://github.com/salemmorelli1/Quantum-Horizon-Microstate-Ensemble`

## 1. Create the empty GitHub repository

On GitHub, create `Quantum-Horizon-Microstate-Ensemble` under the
`salemmorelli1` account. Leave **Add a README**, **.gitignore**, and **license**
unchecked because those files are already included.

## 2. Extract the downloaded ZIP

Extract the repository to:

`C:\Users\salem\GitHub\Quantum-Horizon-Microstate-Ensemble`

Confirm that `README.md`, `pyproject.toml`, `src`, and `docs` are directly inside
that folder. There should not be a second nested copy of the repository.

## 3. Push from Git Bash

```bash
cd "/c/Users/salem/GitHub/Quantum-Horizon-Microstate-Ensemble"

git init
git add .
git commit -m "Initial quantum horizon research framework"
git branch -M main
git remote add origin https://github.com/salemmorelli1/Quantum-Horizon-Microstate-Ensemble.git
git push -u origin main
```

If `origin` already exists, replace the `git remote add` line with:

```bash
git remote set-url origin https://github.com/salemmorelli1/Quantum-Horizon-Microstate-Ensemble.git
```

## 4. Enable the interactive page

Open the repository's **Settings → Pages**. Under **Build and deployment**:

1. choose **Deploy from a branch**;
2. select `main`;
3. select `/docs`;
4. click **Save**.

The site should publish at:

`https://salemmorelli1.github.io/Quantum-Horizon-Microstate-Ensemble/`
