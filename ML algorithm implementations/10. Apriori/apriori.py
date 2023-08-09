from csv import reader
from collections import defaultdict
from itertools import chain, combinations


def powerset(s):
    # Generate all subsets of input set s
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def get_above_min_support(item_set, item_set_list, min_support, global_item_set_with_support):
    freq_item_set = set()
    local_item_set_with_support = defaultdict(int)

    for item in item_set:
        for item_set in item_set_list:
            if item.issubset(item_set):
                global_item_set_with_support[item] += 1
                local_item_set_with_support[item] += 1

    for item, support_count in local_item_set_with_support.items():
        support = float(support_count / len(item_set_list))
        if support > min_support:
            freq_item_set.add(item)
    return freq_item_set


def get_union(item_set, k):
    # Get merging results for all item sets if the merged set has k elements
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == k])


def pruning(candidate_sets, prev_freq_set, k):
    pruned_candidate_sets = candidate_sets.copy()
    for candidate in candidate_sets:
        subsets = combinations(candidate, k)
        for subset in subsets:
            # If any subset of set isn't frequent, the set can't be frequent
            if frozenset(subset) not in prev_freq_set:
                pruned_candidate_sets.remove(candidate)
                break
    return pruned_candidate_sets


def association_rule(freq_item_set, item_set_with_sup, min_confidence):
    rules = []
    for k, item_set in freq_item_set.items():
        for item in item_set:
            subsets = powerset(item)
            for subset in subsets:
                confidence = float(item_set_with_sup[item] / item_set_with_sup[frozenset(subset)])
                if confidence > min_confidence:
                    rules.append([set(subset), set(item.difference(subset)), confidence])
    return rules


def get_single_item_set_from_list(item_set_list):
    temp_item_set = set()
    for item_set in item_set_list:
        for item in item_set:
            temp_item_set.add(frozenset([item]))
    return temp_item_set


def apriori(item_set_list, min_support, min_confidence):
    C1 = get_single_item_set_from_list(item_set_list)
    global_freq_item_set = dict()
    global_item_set_with_sup = defaultdict(int)

    L1 = get_above_min_support(C1, item_set_list, min_support, global_item_set_with_sup)
    current_l_set = L1
    k = 2

    while current_l_set:
        global_freq_item_set[k - 1] = current_l_set
        candidates = get_union(current_l_set, k)
        candidates = pruning(candidates, current_l_set, k - 1)
        current_l_set = get_above_min_support(candidates, item_set_list, min_support, global_item_set_with_sup)
        k += 1

    rules = association_rule(global_freq_item_set, global_item_set_with_sup, min_confidence)
    # Sort based on confidence
    rules.sort(key=lambda x : x[2])
    return global_freq_item_set, rules


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


if __name__ == '__main__':
    item_set_list = load_data_set()
    freq_item_sets, rules = apriori(item_set_list, 0.5, 0.5)
    print(freq_item_sets)
    print(rules)