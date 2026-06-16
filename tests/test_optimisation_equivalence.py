"""
Regression tests: the optimised implementations of f2, f3, predict, and
predict_proba must produce numerically identical results to the original
loop-based code they replaced.

Each test class embeds the original implementation verbatim as a `_ref`
function and asserts exact equality on the same inputs.
"""

import itertools
import random

import numpy as np
import pandas as pd
import pytest

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.algorithms.ids import IDS
from pyids.algorithms.ids_classifier import IDSClassifier, mine_CARs
from pyids.algorithms.ids_objective_function import (
    IDSObjectiveFunction,
    ObjectiveFunctionParameters,
)
from pyids.algorithms.rule_comparator import IDSComparator
from pyids.data_structures.ids_cacher import IDSCacher
from pyids.data_structures.ids_rule import IDSRule
from pyids.data_structures.ids_ruleset import IDSRuleSet


# ---------------------------------------------------------------------------
# Reference implementations — verbatim copies of the pre-optimisation code
# ---------------------------------------------------------------------------

def _f2_ref(solution_set, cacher, quant_dataframe, len_all_rules):
    overlap_intraclass_sum = 0
    for i, r1 in enumerate(solution_set.ruleset):
        for j, r2 in enumerate(solution_set.ruleset):
            if i >= j:
                continue
            if r1.car.consequent.value == r2.car.consequent.value:
                overlap_intraclass_sum += cacher.overlap(r1, r2)
    return quant_dataframe.dataframe.shape[0] * len_all_rules ** 2 - overlap_intraclass_sum


def _f3_ref(solution_set, cacher, quant_dataframe, len_all_rules):
    overlap_interclass_sum = 0
    for i, r1 in enumerate(solution_set.ruleset):
        for j, r2 in enumerate(solution_set.ruleset):
            if i >= j:
                continue
            if r1.car.consequent.value != r2.car.consequent.value:
                overlap_interclass_sum += cacher.overlap(r1, r2)
    return quant_dataframe.dataframe.shape[0] * len_all_rules ** 2 - overlap_interclass_sum


def _predict_ref(clf, quant_dataframe, order_type="f1"):
    sorted_rules = IDSComparator().sort(clf.rules, order_type=order_type)
    predicted_classes = []
    for _, row in quant_dataframe.dataframe.iterrows():
        appended = False
        for rule in sorted_rules:
            antecedent_dict = dict(rule.car.antecedent)
            counter = True
            for name, value in row.items():
                if name in antecedent_dict:
                    counter &= antecedent_dict[name] == value
            if counter:
                _, predicted_class = rule.car.consequent
                predicted_classes.append(predicted_class)
                appended = True
                break
        if not appended:
            predicted_classes.append(clf.default_class)
    return predicted_classes


def _predict_proba_ref(clf, quant_dataframe, order_type="f1"):
    sorted_rules = IDSComparator().sort(clf.rules, order_type=order_type)
    confidences = []
    for _, row in quant_dataframe.dataframe.iterrows():
        appended = False
        for rule in sorted_rules:
            antecedent_dict = dict(rule.car.antecedent)
            counter = True
            for name, value in row.items():
                if name in antecedent_dict:
                    counter &= antecedent_dict[name] == value
            if counter:
                confidences.append(rule.car.confidence)
                appended = True
                break
        if not appended:
            confidences.append(clf.default_class_confidence)
    return confidences


# ---------------------------------------------------------------------------
# Shared fixture: mined rules + cacher built from iris0.csv
# ---------------------------------------------------------------------------

DATA_PATH = "data/iris0.csv"
RULE_CUTOFF = 30
RANDOM_SEED = 42


@pytest.fixture(scope="module")
def iris_setup():
    df = pd.read_csv(DATA_PATH)
    quant_df = QuantitativeDataFrame(df)

    cars = mine_CARs(df, rule_cutoff=RULE_CUTOFF, random_seed=RANDOM_SEED)
    ids_rules = list(map(IDSRule, cars))
    all_rules = IDSRuleSet(ids_rules)

    cacher = IDSCacher()
    cacher.calculate_overlap(all_rules, quant_df)

    params = ObjectiveFunctionParameters()
    params.params["all_rules"] = all_rules
    params.params["len_all_rules"] = len(all_rules)
    params.params["quant_dataframe"] = quant_df
    params.params["lambda_array"] = 7 * [1]

    obj_fn = IDSObjectiveFunction(objective_func_params=params, cacher=cacher)

    return {
        "df": df,
        "quant_df": quant_df,
        "all_rules": all_rules,
        "cacher": cacher,
        "obj_fn": obj_fn,
        "n": len(all_rules),
    }


# ---------------------------------------------------------------------------
# f2 / f3 equivalence
# ---------------------------------------------------------------------------

class TestF2F3Equivalence:

    def _assert_equal(self, subset_rules, setup):
        rs = IDSRuleSet(set(subset_rules))
        obj_fn = setup["obj_fn"]
        cacher = setup["cacher"]
        quant_df = setup["quant_df"]
        n = setup["n"]

        assert obj_fn.f2(rs) == _f2_ref(rs, cacher, quant_df, n), \
            f"f2 mismatch for subset of size {len(rs)}"
        assert obj_fn.f3(rs) == _f3_ref(rs, cacher, quant_df, n), \
            f"f3 mismatch for subset of size {len(rs)}"

    def test_empty_set(self, iris_setup):
        self._assert_equal(set(), iris_setup)

    def test_all_singletons(self, iris_setup):
        for rule in iris_setup["all_rules"].ruleset:
            self._assert_equal({rule}, iris_setup)

    def test_all_pairs(self, iris_setup):
        rules = list(iris_setup["all_rules"].ruleset)
        for r1, r2 in itertools.combinations(rules, 2):
            self._assert_equal({r1, r2}, iris_setup)

    def test_full_ruleset(self, iris_setup):
        self._assert_equal(iris_setup["all_rules"].ruleset, iris_setup)

    def test_random_subsets(self, iris_setup):
        rng = random.Random(RANDOM_SEED)
        rules = list(iris_setup["all_rules"].ruleset)
        for size in [3, 5, 8, 12]:
            for _ in range(5):
                subset = rng.sample(rules, min(size, len(rules)))
                self._assert_equal(subset, iris_setup)


# ---------------------------------------------------------------------------
# predict / predict_proba equivalence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_ids(iris_setup):
    quant_df = iris_setup["quant_df"]
    cars = mine_CARs(iris_setup["df"], rule_cutoff=RULE_CUTOFF, random_seed=RANDOM_SEED)

    ids = IDS(algorithm="SLS")
    ids.fit(
        quant_dataframe=quant_df,
        class_association_rules=cars,
        lambda_array=7 * [1],
        random_seed=RANDOM_SEED,
    )
    return ids, quant_df


class TestPredictEquivalence:

    def test_predict_on_training_data(self, fitted_ids):
        ids, quant_df = fitted_ids
        clf = ids.clf

        new_result = clf.predict(quant_df)
        ref_result = _predict_ref(clf, quant_df)

        assert new_result == ref_result

    def test_predict_proba_on_training_data(self, fitted_ids):
        ids, quant_df = fitted_ids
        clf = ids.clf

        new_result = clf.predict_proba(quant_df)
        ref_result = _predict_proba_ref(clf, quant_df)

        assert len(new_result) == len(ref_result)
        assert all(abs(a - b) < 1e-12 for a, b in zip(new_result, ref_result))

    def test_predict_on_held_out_data(self, iris_setup, fitted_ids):
        ids, _ = fitted_ids
        clf = ids.clf

        # Use the second half of the dataset as a held-out set
        df = iris_setup["df"]
        df_test = df.iloc[len(df) // 2:].reset_index(drop=True)
        quant_df_test = QuantitativeDataFrame(df_test)

        new_result = clf.predict(quant_df_test)
        ref_result = _predict_ref(clf, quant_df_test)

        assert new_result == ref_result

    def test_predict_proba_on_held_out_data(self, iris_setup, fitted_ids):
        ids, _ = fitted_ids
        clf = ids.clf

        df = iris_setup["df"]
        df_test = df.iloc[len(df) // 2:].reset_index(drop=True)
        quant_df_test = QuantitativeDataFrame(df_test)

        new_result = clf.predict_proba(quant_df_test)
        ref_result = _predict_proba_ref(clf, quant_df_test)

        assert len(new_result) == len(ref_result)
        assert all(abs(a - b) < 1e-12 for a, b in zip(new_result, ref_result))

    def test_predict_order_types(self, fitted_ids):
        ids, quant_df = fitted_ids
        clf = ids.clf

        for order_type in ["f1", "cba"]:
            new_result = clf.predict(quant_df, order_type=order_type)
            ref_result = _predict_ref(clf, quant_df, order_type=order_type)
            assert new_result == ref_result, \
                f"predict mismatch for order_type={order_type!r}"

    def test_predict_default_class_for_uncovered_rows(self, iris_setup):
        """Rows that no rule fires on should receive the default class."""
        df = iris_setup["df"]
        quant_df = iris_setup["quant_df"]
        cars = mine_CARs(df, rule_cutoff=RULE_CUTOFF, random_seed=RANDOM_SEED)

        # Fit with n_select=1 so only one rule is selected — maximises uncovered rows.
        ids = IDS(algorithm="SLS", n_select=1)
        ids.fit(
            quant_dataframe=quant_df,
            class_association_rules=cars,
            lambda_array=7 * [1],
            random_seed=RANDOM_SEED,
        )
        clf = ids.clf

        new_result = clf.predict(quant_df)
        ref_result = _predict_ref(clf, quant_df)

        assert new_result == ref_result
