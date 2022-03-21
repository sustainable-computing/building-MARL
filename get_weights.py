from glob import glob
import shutil
import os

library_location = "policy_library"
os.makedirs(library_location, exist_ok=True)
name_convention = {"0.1": "1e-1", "1.0": "1e0", "10.0": "1e1"}

for path in glob("PPO_weights/**"):
    prefix = path[-3:]
    if prefix[0] == "7":
        continue

    for i in range(5):
        shutil.copy(f"{path}/agent_{i}.pth", f"{library_location}/{prefix}_{i}.pth")

    for subpath in glob(f"{path}/**/"):
        for i in range(5):
            shutil.copy(f"{subpath}/agent_{i}.pth", f"{library_location}/{prefix}_{i}_{name_convention[os.path.basename(subpath[:-1])]}.pth")
