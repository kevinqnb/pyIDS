# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (preferred)
pip install -e .
# or with uv
uv pip install -e .

# Run the main entry point
python main.py

# Run a use-case script
python scripts/use_case/ids_model_building.py
```

There is no test suite or linter configured in this project.

## Architecture

pyIDS implements the Interpretable Decision Sets (IDS) algorithm (Lakkaraju et al., 2016). The algorithm selects a small set of classification rules that jointly maximize a 7-component interpretability/accuracy objective.

### Data flow

1. **Rule mining** (`pyids/algorithms/ids_classifier.py:mine_CARs`): Uses `pyarc` to mine Class Association Rules (CARs) from a pandas DataFrame via FP-growth. Returns a list of `ClassAssocationRule` objects.

2. **Data container**: All training/scoring calls require a `QuantitativeDataFrame` (from `pyarc.qcba.data_structures`), not a raw DataFrame. Wrap with `QuantitativeDataFrame(df)`.

3. **Fitting** (`IDS.fit` in `pyids/algorithms/ids.py`): Wraps CARs in `IDSRule` objects → builds `IDSRuleSet` → pre-computes all pairwise rule overlaps via `IDSCacher` → runs one of four optimizers to find the best subset → optionally trims to `n_select` rules via iterative backward elimination.

4. **Prediction** (`IDSClassifier` in `pyids/algorithms/ids_classifier.py`): Rules are sorted by F1 score (or another `order_type`), then the first matching rule for each row is used. Unmatched rows use a default class (majority among uncovered training samples by default).

### Key classes

- **`IDS`** (`pyids/algorithms/ids.py`): Public API. Parameters: `algorithm` (SLS/DLS/DUSM/RUSM), `n_select` (max rules, trimmed via backward elimination after optimization).

- **`IDSObjectiveFunction`** (`pyids/algorithms/ids_objective_function.py`): Weighted sum of 7 sub-objectives f0–f6 (fewer rules, shorter rules, less intra-class overlap, more inter-class overlap, class coverage, fewer incorrect covers, more correct covers). The `lambda_array` (length 7) controls their weights. Setting a lambda to 0 skips that term.

- **`IDSCacher`** (`pyids/data_structures/ids_cacher.py`): Pre-computes all pairwise rule overlaps and per-rule covers at `fit` time. Must be reused across optimizer calls to avoid redundant computation. Note: logger was removed to allow pickling.

- **`IDSRule`** (`pyids/data_structures/ids_rule.py`): Wraps `ClassAssocationRule` with cached cover masks (`cover`, `correct_cover`, `incorrect_cover`, `rule_cover`) and an F1 score. Cache is populated by `calculate_cover(quant_dataframe)`.

- **Optimizers** (`pyids/algorithms/optimizers/`): SLS (Stochastic Local Search — default, recommended), DLS (Deterministic), DUSM/RUSM (Unconstrained Submodular Maximization variants).

### Lambda hyperparameter optimization

Three strategies in `pyids/model_selection/`:
- **`CoordinateAscent`**: Ternary search over each λᵢ in sequence; auto-extends search range if optimum is near boundary. Takes a user-supplied `func(lambda_dict) -> float`.
- **`RandomSearch`** / **`GridSearch`**: Simpler alternatives.

### Interpretability metrics

`pyids/model_selection/metrics.py` provides `calculate_ruleset_statistics` (and individual functions) for fraction overlap, fraction uncovered, average rule width, fraction of classes covered, and ruleset length. Accessible via `ids.score_interpretability_metrics(quant_dataframe)`.
