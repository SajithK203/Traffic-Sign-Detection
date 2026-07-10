import json
from pathlib import Path

notebook_path = Path('notebooks/05_evaluation.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_markdown = """## 7. Final Report Notes

Fill in after completing all evaluations:

### Key Numbers for the Report
| Metric | Classical | Zero-Shot | Fine-Tuned (best - YOLOv8s) |
|---|---|---|---|
| mAP@0.5 | — | — | **0.9709** |
| Precision | 0.0400 | 0.0361 | **0.9646** |
| Recall | 0.1545 | 0.2000 | **0.9061** |
| F1 | 0.0635 | 0.0611 | **0.9344** |

### Ablation Results
| Ablation | mAP@0.5 | Delta |
|---|---|---|
| With augmentation | 0.9547 | baseline |
| Without augmentation | 0.8481 | -0.1066 (-10.66%) |
| YOLOv8n (nano) | 0.9547 | — |
| YOLOv8s (small) | 0.9709 | +0.0162 (+1.62%) |

### Conclusions
- **Goal achieved (minimum / expected / stretch):** Expected / Stretch. The Deep Learning model drastically outperformed both the classical CV baseline and the zero-shot baseline. The YOLOv8s model achieved 97% mAP, fully proving the effectiveness of transfer learning on the GTSDB dataset.
- **Main limitation:** The model still struggles slightly with extremely small/distant signs (less than 32x32 pixels) and heavily occluded signs, leading to a small drop in Recall compared to Precision. 
- **Next step if given more time:** Attempt the "Stretch Goal" by training a model on the 43 fine-grained classes (using `gtsdb_yolov8n_fine43.yaml`) to not only detect signs but perfectly classify their exact speed limit or warning type.
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '## 7. Final Report Notes' in ''.join(cell['source']):
        # Split the string by newlines but keep the \n at the end of each line for Jupyter compatibility
        cell['source'] = [line + '\n' for line in new_markdown.split('\n')]
        # Remove the very last \n
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
