# Note

This folder contains three images for a mixed-inference demo: a cat, a dog, and a bear. The bear represents an unknown species. The demo shows that an unsupported species is handled as a per-image failure, while the rest of the batch continues processing.

The samples correspond to `data/raw/cat/1.jpg`, `data/raw/dog/1.jpg`, and `data/unknown/bear_1.jpg`.

To keep API usage low, run:

```bash
python main.py --input-dir inference_data --limit 3
```

The folder currently contains exactly three images, so `--limit 3` processes the full demo set. The cat and dog can continue through the generation pipeline; the bear is marked unsupported at Part B if Part A identifies it as an unknown species.
