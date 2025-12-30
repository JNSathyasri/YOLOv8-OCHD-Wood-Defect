from datasets import load_dataset
import pprint

dataset = load_dataset("iluvvatar/wood_surface_defects", split="train")

sample = dataset[0]

print("Sample keys:")
print(sample.keys())

print("\nObjects type:")
print(type(sample["objects"]))

print("\nFirst object:")
pprint.pprint(sample["objects"][0])
