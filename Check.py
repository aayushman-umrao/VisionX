"""
Quick diagnostic: inspect the labelmap.txt file to check for indexing issues
that could cause misclassification (e.g. pen -> "cell phone").
"""

with open("models/labelmap.txt", "r") as f:
    lines = [line.rstrip("\n").rstrip("\r") for line in f.readlines()]

print("Total lines:", len(lines))
print()
print("First 5 lines (with index numbers):")
for i, line in enumerate(lines[:5]):
    print(f"  [{i}] -> '{line}'")

print()
print("Lines 60 to 70 (with index numbers):")
for i, line in enumerate(lines[60:70], start=60):
    print(f"  [{i}] -> '{line}'")

print()
print("Does line [0] equal '???' ?", lines[0] == "???")