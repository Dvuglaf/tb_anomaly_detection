"""
    Converting .yaml format labels to .csv format labels.

    Usage: $python convert_range_labels_2_binary_encoding.py path_to_input_file.yaml
"""

import re
import sys
import numpy as np
import pandas as pd


if len(sys.argv) != 2:
    print("\nUsage example: python convert_range_labels_2_binary_encoding.py IMG_1769.yaml\n")
    exit()

file_name = sys.argv[1]
print(f"Converting {file_name}")
print()

labels_yaml = open(file_name, "r").read()

# extract from yaml class names, labels and ranges
class_name_label_mapping = {}
class_label_ranges_mapping = {}
max_frame = 0
for class_label, line in enumerate(labels_yaml.split("\n")):
    line = line.strip()
    match_result = re.match('(.+):(.+)', line)
    if match_result:
        class_name = match_result.group(1)
        class_name_label_mapping[class_name] = class_label
        class_ranges = match_result.group(2).strip().strip(",").split(",")
        for class_range in class_ranges:
            class_range = class_range.strip()
            start, stop = class_range.split("-")
            start, stop = int(start), int(stop)
            old_ranges = class_label_ranges_mapping.get(class_label, [])
            class_label_ranges_mapping[class_label] = old_ranges + [(start, stop)]
            max_frame = max(max_frame, stop)
    else:
        match_result = re.match('(.+):', line)
        class_name = match_result.group(1)
        class_name_label_mapping[class_name] = class_label
        class_label_ranges_mapping[class_label] = []

class_name_label_mapping_reverse = {v: k for k, v in class_name_label_mapping.items()}

print(f"Max labeled frame index: {max_frame}. Assuming indexes from 1.")
print(f"Class count: {len(class_name_label_mapping.keys())}")
print()

# convert to binary encoding
binary_encoding = np.full((max_frame, len(class_name_label_mapping.keys())), fill_value=False)

class_names = []
for class_label, class_ranges in class_label_ranges_mapping.items():
    class_names.append(class_name_label_mapping_reverse[class_label])
    for start, stop in class_ranges:
        start = start - 1
        stop = stop - 1
        binary_encoding[start:stop, class_label] = True

df_binary_encoding = pd.DataFrame(data=binary_encoding, columns=class_names)

csv_file_name = f"{file_name.replace('.yaml', '')}.csv"
excel_file_name = f"{file_name.replace('.yaml', '')}.xlsx"
df_binary_encoding.to_csv(csv_file_name, index=False)
print(f"Saved {csv_file_name}")