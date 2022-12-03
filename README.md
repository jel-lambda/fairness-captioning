# fairness-captioning
fairness-captioning

# 0. Dataset preprocessing
---

## Minimum settings to run show-attend-and-tell

Download the COCO dataset training and validation images. Put COCO dataset split JSON file from [Deep Visual-Semantic Alignments](https://cs.stanford.edu/people/karpathy/deepimagesent/) in `data/coco/`. It should be named `dataset.json`

Run the preprocessing to create the needed JSON files:
```
python generate_json_data.py
```