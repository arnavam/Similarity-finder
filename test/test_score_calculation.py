from collections import Counter

def calculate_overall_match(values):
    if not values:
        return 0.0
    
    counts = Counter(values)
    max_freq = max(counts.values())
    
    if max_freq == 1:
        # All unique, take max
        return max(values)
    else:
        # Take max of the modes
        modes = [k for k, v in counts.items() if v == max_freq]
        return max(modes)

# Test Cases
test_cases = [
    ([0.1, 0.2, 0.3], 0.3, "All unique"),
    ([0.5, 0.5, 0.2], 0.5, "Single mode"),
    ([0.6, 0.6, 0.7, 0.7, 0.1], 0.7, "Multiple modes (tie-break max)"),
    ([0.8], 0.8, "Single item"),
    ([], 0.0, "Empty"),
    ([0.1, 0.2, 0.2, 0.3, 0.3, 0.3], 0.3, "Clear mode"),
]

print("Running Logic Verification...")
failed = False
for i, (input_vals, expected, desc) in enumerate(test_cases):
    result = calculate_overall_match(input_vals)
    if abs(result - expected) < 1e-9:
        print(f"✅ Case {i+1} ({desc}): PASS (Got {result})")
    else:
        print(f"❌ Case {i+1} ({desc}): FAIL (Expected {expected}, Got {result})")
        failed = True

if not failed:
    print("\nAll tests passed!")
else:
    print("\nSome tests failed.")
