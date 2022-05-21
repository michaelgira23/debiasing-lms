def filter_combinations(criteria):
    def filter_combination_func(check):
        available_keys = set(check.keys())

        for banned_combination in criteria:
            if set(banned_combination.keys()) <= available_keys:
                is_different = False
                for combination_key in banned_combination:
                    if banned_combination[combination_key] != check[combination_key]:
                        is_different = True
                        break
                # Criteria exists in the combination in question
                if not is_different:
                    return False
        return True

    return filter_combination_func


def generate_combinations(reference_variables, keys, disallow_combinations):
    if not len(keys):
        return [{}]

    tweak_key = keys.pop(0)
    suffixes = generate_combinations(
        reference_variables, keys, disallow_combinations)

    total_combinations = []
    for suffix in suffixes:
        for value in reference_variables[tweak_key]:
            new_suffix = suffix.copy()
            new_suffix[tweak_key] = value
            total_combinations.append(new_suffix)

    # Filter out blacklist of combinations
    # Without filter: 2560 combinations
    # With filter: 1080 combinations (~58% reduction)
    return list(filter(filter_combinations(disallow_combinations), total_combinations))
