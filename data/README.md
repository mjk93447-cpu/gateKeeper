# Data policy

Raw production images are sensitive and large. They are intentionally excluded
from Git and from public release artifacts.

Store local data using the following layout:

```text
data/
  raw/
  processed/
  manifests/
  feedback/
```

Dataset versions are defined by a manifest containing relative paths, image
hashes, panel/lot/recipe metadata, lighting slices, and split membership. Frames
from the same panel or continuous burst must remain in one split.
