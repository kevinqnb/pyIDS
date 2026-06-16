import numpy as np


class IDSCacher:

    def __init__(self):
        self.overlap_cache = dict()
        self.rule_index = {}
        self.overlap_matrix = None
        self.same_class_matrix = None

    def overlap(self, rule1, rule2):
        return self.overlap_cache[repr(rule1) + repr(rule2)]

    def calculate_overlap(self, all_rules, quant_dataframe):
        rules_list = list(all_rules.ruleset)
        n = len(rules_list)

        self.rule_index = {rule: i for i, rule in enumerate(rules_list)}
        self.overlap_matrix = np.zeros((n, n), dtype=np.int64)
        self.same_class_matrix = np.zeros((n, n), dtype=bool)

        for rule in rules_list:
            rule.calculate_cover(quant_dataframe)

        for i, rule_i in enumerate(rules_list):
            for j, rule_j in enumerate(rules_list):
                overlap_len = int(np.sum(rule_i.rule_overlap(rule_j, quant_dataframe)))

                self.overlap_cache[repr(rule_i) + repr(rule_j)] = overlap_len
                self.overlap_matrix[i, j] = overlap_len
                self.same_class_matrix[i, j] = (
                    rule_i.car.consequent.value == rule_j.car.consequent.value
                )
