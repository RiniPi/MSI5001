# MSI5001: agent brief

Group 10's coursework for MSI5001: classifying book synopses into 10 genres (fantasy, science, crime, history, horror, thriller, psychology, romance, sports, travel). Three models are compared: TF-IDF + Logistic Regression + SMOTE, fastText + XGBoost, and fine-tuned DistilBERT. `README.md` is the write-up, so read it before doing anything.

## At the start of every session
1. Bring in the latest `main`: `git fetch origin` and merge `origin/main` into your branch, or branch from it.
2. Read `README.md`, then the "Open questions" section below. Don't decide those yourself.
3. Tell the user what you plan to change before moving, renaming or deleting files.

## Repo layout
- `README.md`, `images/`: the write-up and its figures. **Both stay at the top level**, because the README's image links are relative (`images/NN_description.png`).
- Top-level notebook and CSVs: files the README's "Files" table names. Keep them at the top level.
- `results/logistic-regression/`, `results/distilbert/`: charts, prediction CSVs and error analyses per model.
- `models/logistic-regression/`: saved model, vectoriser and label mapping (`.pkl`).
- `docs/`: slides and Word reports.

## While working
- The user works on Windows in Command Prompt: give commands in that form (`dir`, `move`, `\` paths, quoted names with spaces).
- Notebooks run in Google Colab (CPU for Logistic Regression and XGBoost, T4 GPU for DistilBERT). `data.csv` is uploaded when the notebook asks and is not in the repo.
- Notebooks save outputs by bare filename in Colab's working folder. Move new outputs into the right `results/` or `models/` folder before committing.
- New README figures go in `images/` with the next number (`16_…png`) and are linked as `images/16_….png`. Names are case-sensitive on GitHub.
- When the task is reorganising, don't change file contents. Move files with `git mv` so history is kept.
- Don't commit zip files; extract them and commit the contents. Don't create copies like `name (1).xlsx` or `README (2).md`.
- Files over 100 MB can't be pushed (over 50 MB gets a warning). `docs/MSI5001 - Group 10_.pptx` is already about 63 MB.
- Commit as the personal account `RiniPi`, not the work account `rini-adamata`.

## Open questions (ask the user)
- `results/distilbert/error_rate_by_genre_tfidf_smote.png` is byte-identical to the Logistic Regression copy; it looks misfiled.
- `results/logistic-regression/misclassified_samples.xlsx` and `misclassified_samples(1).xlsx` differ; which is final is unknown.
- The README lists `[FINAL]_distilBERT_8epochs.ipynb` and `[FINAL] XGBOOST.ipynb`, but neither is in the repo.

## At the end of every session
- If a change affects the README (new figure, moved file it names), update the README in the same PR.
- Remove a question from "Open questions" once the user has answered it.
- Push your branch and open or refresh the PR to `main`.
