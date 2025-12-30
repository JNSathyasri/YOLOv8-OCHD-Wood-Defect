from datasets import load_dataset

dataset = load_dataset("iluvvatar/wood_surface_defects")

print(dataset)
print("\nSplits:")
for split in dataset:
    print(split, len(dataset[split]))

print("\nSample keys:")
print(dataset[list(dataset.keys())[0]][0].keys())