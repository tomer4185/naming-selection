# Human vs LLM Naming Selection

## Preprocess
1. Copy the desired URLs into a text file named `repos.txt`.
2. Run `github_extractor` with the relevant URLs — this will create a `downloaded_repos` folder containing all `.py` files.
3. Run `repo_anonymizer` to extract the required information into `anonymized_variables.json`.
4. For statistics, run `variable_extractor` — this generates `variables_detailed.json` and `analysis.json`.

## LLM
Run `llm_renamer` to replace variables in the extracted code using the LLM.

## Evaluation
Evaluation scripts can be found in the `evaluation` package.
