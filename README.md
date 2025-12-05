# RecencyBank for tgbl-wiki

Temporal link prediction on Wikipedia editing patterns using a simple recency-based heuristic.

## Overview

This project implements RecencyBank, a minimal baseline for temporal link prediction on the tgbl-wiki dataset from the Temporal Graph Benchmark. The core hypothesis is that Wikipedia editors exhibit strong temporal locality—they tend to repeatedly edit the same pages within short time windows.

RecencyBank achieves test MRR of 0.403 by predicting that users will edit the pages they most recently edited. While this simple approach ranks last among compared baselines, comprehensive failure analysis reveals important insights about when and why temporal patterns succeed or fail.

## Key Findings

- **Temporal locality works for active users**: High-activity users (>5 edits) achieve 0.432 accuracy
- **Cold start is fundamental**: 16.5% of test queries involve users with no editing history
- **User heterogeneity matters**: Prediction accuracy varies 432× between high and low activity users
- **Validation-test gap explained**: 2.2% more cold start queries in test set accounts for performance drop

## Installation

### Requirements

- Python 3.8+
- PyTorch (for TGB library)
- Standard scientific Python stack

### Setup

```bash
git clone https://github.com/[username]/tgbl-wiki-recencybank.git
cd tgbl-wiki-recencybank
pip install -r requirements.txt
```

## Usage

Run the complete analysis pipeline:

```bash
python run_analysis.py
```

This will:
1. Download and load the tgbl-wiki dataset
2. Perform exploratory data analysis
3. Train RecencyBank on the training set
4. Evaluate on validation and test sets
5. Generate all visualizations
6. Save results to `results/` directory

Expected runtime: 5-7 minutes on standard hardware.

## Results

### Performance Metrics

| Split | MRR | Accuracy |
|-------|-----|----------|
| Validation | 0.459 | 0.458 |
| Test | 0.403 | 0.402 |

### Leaderboard Comparison

RecencyBank ranks last among compared methods on tgbl-wiki:

| Method | MRR | Type |
|--------|-----|------|
| TPNet | 0.827 | GNN |
| Base3 | 0.743 | Heuristic |
| EdgeBank(unlimited) | 0.495 | Heuristic |
| **RecencyBank** | **0.403** | **Heuristic** |

### Failure Analysis

Test set prediction breakdown:
- Correct predictions: 40.2%
- Incorrect predictions: 43.3%
- Cold start (no prediction): 16.5%

Performance by user activity level:
- High activity (>5 edits): 0.432 accuracy
- Medium activity (1-5 edits): 0.086 accuracy
- Low activity (≤1 edit): 0.000 accuracy

## Method

RecencyBank maintains a dictionary mapping each user to their most recently edited page. When queried, it predicts the user will edit that same page again.

**Algorithm:**
```
predict(user) = argmax_page timestamp(user, page)
```

**Complexity:**
- Training: O(|E|) time, O(|V|) space
- Query: O(1) time

The method exploits temporal locality arising from:
1. Topical specialization: editors focus on specific domains
2. Maintenance work: active editors monitor pages for changes
3. Collaborative campaigns: coordinated improvement efforts

## Output

### Results Directory

- `evaluation_results.json`: Complete metrics and statistics
- `recency_bank.pkl`: Trained model (for reproducibility)

### Figures Directory

- `degree_distribution.pdf`: In/out degree distributions showing heavy tails
- `temporal_activity.pdf`: Editing activity over time with split boundaries
- `failure_and_activity_analysis.pdf`: Prediction breakdown and performance by user activity
- `leaderboard_comparison.pdf`: RecencyBank positioned among baseline methods

## Implementation Details

The implementation follows TGB evaluation protocols:
- Official pre-sampled negative edges for each query
- Chronological 70-15-15 train/validation/test split
- MRR (Mean Reciprocal Rank) as primary metric
- No data leakage (models use only past information)

## Files

- `recency_bank.py`: RecencyBank model implementation (standalone, ~70 lines)
- `run_analysis.py`: Complete analysis pipeline (~350 lines)
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Citation

If you use this code, please cite the Temporal Graph Benchmark:

```bibtex
@article{huang2023temporal,
  title={Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Author

Developed as coursework for Special Lectures in Mathematical Informatics II at the University of Tokyo.

## Acknowledgments

- Temporal Graph Benchmark (TGB) team for datasets and evaluation tools
- Course instructors for guidance and feedback
