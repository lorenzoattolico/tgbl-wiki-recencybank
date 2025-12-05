"""
TGB-Wiki RecencyBank Analysis

Complete pipeline for temporal link prediction on tgbl-wiki dataset.
Includes dataset exploration, model training, evaluation, and visualization.

Author: [Your Name]
Course: Special Lectures in Mathematical Informatics II
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

from recency_bank import RecencyBank


# Configuration
RANDOM_SEED = 42
DATASET_NAME = "tgbl-wiki"
ROOT_DIR = "datasets"
OUTPUT_DIR = "results"
FIGURES_DIR = "figures"

np.random.seed(RANDOM_SEED)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(FIGURES_DIR).mkdir(exist_ok=True)

# Plotting configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def load_and_explore_dataset():
    """Load tgbl-wiki dataset and perform exploratory analysis."""
    print_section("Loading Dataset")
    
    dataset = LinkPropPredDataset(name=DATASET_NAME, root=ROOT_DIR)
    
    num_nodes = dataset.num_nodes
    num_edges = dataset.num_edges
    edge_list = dataset.full_data
    
    sources = edge_list['sources']
    destinations = edge_list['destinations']
    timestamps = edge_list['timestamps']
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")
    print(f"  Temporal span: {timestamps[-1] - timestamps[0]:,} time units")
    
    # Get splits
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    train_data = {
        'sources': sources[train_mask],
        'destinations': destinations[train_mask],
        'timestamps': timestamps[train_mask]
    }
    
    val_data = {
        'sources': sources[val_mask],
        'destinations': destinations[val_mask],
        'timestamps': timestamps[val_mask]
    }
    
    test_data = {
        'sources': sources[test_mask],
        'destinations': destinations[test_mask],
        'timestamps': timestamps[test_mask]
    }
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_data['sources']):,} ({len(train_data['sources'])/num_edges*100:.1f}%)")
    print(f"  Val: {len(val_data['sources']):,} ({len(val_data['sources'])/num_edges*100:.1f}%)")
    print(f"  Test: {len(test_data['sources']):,} ({len(test_data['sources'])/num_edges*100:.1f}%)")
    
    # Compute degree statistics
    out_degree = Counter(sources)
    in_degree = Counter(destinations)
    
    print(f"\nDegree Statistics:")
    print(f"  Out-degree mean: {np.mean(list(out_degree.values())):.2f}")
    print(f"  Out-degree max: {max(out_degree.values()):,}")
    print(f"  In-degree mean: {np.mean(list(in_degree.values())):.2f}")
    print(f"  In-degree max: {max(in_degree.values()):,}")
    
    # User activity stratification
    out_degrees_list = list(out_degree.values())
    low_threshold = np.percentile(out_degrees_list, 33)
    high_threshold = np.percentile(out_degrees_list, 67)
    
    return {
        'dataset': dataset,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'out_degree': out_degree,
        'in_degree': in_degree,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'sources': sources,
        'destinations': destinations,
        'timestamps': timestamps,
    }


def create_visualizations(data_dict):
    """Generate all visualizations for the report."""
    print_section("Generating Visualizations")
    
    out_degree = data_dict['out_degree']
    in_degree = data_dict['in_degree']
    low_threshold = data_dict['low_threshold']
    high_threshold = data_dict['high_threshold']
    timestamps = data_dict['timestamps']
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    
    # Figure 1: Degree Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    out_degrees = list(out_degree.values())
    axes[0].hist(out_degrees, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].set_xlabel('Out-degree (edits per user)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Out-degree Distribution', fontsize=13, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(low_threshold, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axvline(high_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    in_degrees = list(in_degree.values())
    axes[1].hist(in_degrees, bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
    axes[1].set_xlabel('In-degree (edits per page)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('In-degree Distribution', fontsize=13, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/degree_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: degree_distribution.pdf")
    
    # Figure 2: Temporal Activity
    fig, ax = plt.subplots(figsize=(12, 5))
    
    time_window = (timestamps[-1] - timestamps[0]) // 50
    time_bins = np.arange(timestamps[0], timestamps[-1] + time_window, time_window)
    edge_counts, _ = np.histogram(timestamps, bins=time_bins)
    
    ax.plot(time_bins[:-1], edge_counts, linewidth=2, color='#2c3e50')
    ax.fill_between(time_bins[:-1], edge_counts, alpha=0.3, color='#2c3e50')
    
    train_end = train_data['timestamps'][-1]
    val_end = val_data['timestamps'][-1]
    ax.axvline(train_end, color='blue', linestyle='--', linewidth=2, label='Train/Val', alpha=0.7)
    ax.axvline(val_end, color='red', linestyle='--', linewidth=2, label='Val/Test', alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.set_title('Temporal Activity Pattern', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/temporal_activity.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: temporal_activity.pdf")


def train_recencybank(train_data):
    """Train RecencyBank on training data."""
    print_section("Training RecencyBank")
    
    recency_bank = RecencyBank()
    
    for i in range(len(train_data['sources'])):
        src = train_data['sources'][i]
        dst = train_data['destinations'][i]
        ts = train_data['timestamps'][i]
        recency_bank.update(src, dst, ts)
    
    stats = recency_bank.get_statistics()
    print(f"Trained on {stats['num_updates']:,} edges")
    print(f"Tracking {stats['num_sources']:,} sources ({stats['num_sources']/9227*100:.1f}% coverage)")
    
    with open(f"{OUTPUT_DIR}/recency_bank.pkl", 'wb') as f:
        pickle.dump(recency_bank, f)
    
    return recency_bank


def evaluate_model(data, neg_samples, model, out_degree, low_threshold, high_threshold, split_name="test"):
    """Evaluate RecencyBank with failure analysis and activity stratification."""
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    
    y_pred_pos = []
    y_pred_neg = []
    matched = 0
    
    correct_count = 0
    incorrect_count = 0
    no_pred_count = 0
    
    activity_predictions = {'low': [], 'medium': [], 'high': []}
    
    for i in range(len(sources)):
        src = sources[i]
        dst = destinations[i]
        ts = timestamps[i]
        
        query_key = (src, dst, ts)
        if query_key not in neg_samples:
            continue
        
        neg_dsts = neg_samples[query_key]
        matched += 1
        
        pred_dst = model.query(src)
        
        pos_score = 1.0 if pred_dst == dst else 0.0
        neg_scores = [1.0 if pred_dst == neg_dst else 0.0 for neg_dst in neg_dsts]
        
        y_pred_pos.append(pos_score)
        y_pred_neg.append(neg_scores)
        
        # Failure tracking
        if pred_dst is None:
            no_pred_count += 1
        elif pred_dst == dst:
            correct_count += 1
        else:
            incorrect_count += 1
        
        # Activity stratification
        user_activity = out_degree.get(src, 0)
        if user_activity <= low_threshold:
            activity_predictions['low'].append(pos_score)
        elif user_activity <= high_threshold:
            activity_predictions['medium'].append(pos_score)
        else:
            activity_predictions['high'].append(pos_score)
    
    # Convert to arrays
    y_pred_pos = np.array(y_pred_pos, dtype=np.float64)
    
    max_len = max(len(negs) for negs in y_pred_neg)
    y_pred_neg_array = np.full((len(y_pred_neg), max_len), -np.inf, dtype=np.float64)
    
    for i, negs in enumerate(y_pred_neg):
        y_pred_neg_array[i, :len(negs)] = negs
    
    # Evaluate
    evaluator = Evaluator(name=DATASET_NAME)
    
    input_dict = {
        'y_pred_pos': y_pred_pos,
        'y_pred_neg': y_pred_neg_array,
        'eval_metric': ['mrr', 'hits@']
    }
    
    metrics = evaluator.eval(input_dict)
    
    # Activity accuracy
    activity_acc = {}
    for level, scores in activity_predictions.items():
        if len(scores) > 0:
            activity_acc[level] = {
                'accuracy': float(np.mean(scores)),
                'count': len(scores)
            }
        else:
            activity_acc[level] = {'accuracy': 0.0, 'count': 0}
    
    return {
        'metrics': metrics,
        'matched': matched,
        'failure_stats': {
            'correct': correct_count,
            'incorrect': incorrect_count,
            'no_prediction': no_pred_count,
            'accuracy': correct_count / matched if matched > 0 else 0,
        },
        'activity_accuracy': activity_acc,
    }


def create_analysis_visualizations(test_results, val_results):
    """Create failure analysis and leaderboard comparison figures."""
    print_section("Creating Analysis Visualizations")
    
    # Figure 3: Failure Analysis and Activity Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Failure pie chart
    categories = ['Correct', 'Incorrect', 'No Prediction']
    counts = [
        test_results['failure_stats']['correct'],
        test_results['failure_stats']['incorrect'],
        test_results['failure_stats']['no_prediction']
    ]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = axes[0].pie(counts, labels=categories, colors=colors,
                                             autopct='%1.1f%%', startangle=90,
                                             textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    axes[0].set_title('Prediction Breakdown (Test Set)', fontsize=13, fontweight='bold')
    
    # Activity bar chart
    activity_levels = ['Low', 'Medium', 'High']
    test_accs = [
        test_results['activity_accuracy']['low']['accuracy'],
        test_results['activity_accuracy']['medium']['accuracy'],
        test_results['activity_accuracy']['high']['accuracy']
    ]
    test_counts = [
        test_results['activity_accuracy']['low']['count'],
        test_results['activity_accuracy']['medium']['count'],
        test_results['activity_accuracy']['high']['count']
    ]
    
    x = np.arange(len(activity_levels))
    bars = axes[1].bar(x, test_accs, color=['#e74c3c', '#f39c12', '#2ecc71'], 
                        edgecolor='black', linewidth=1.5)
    
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_xlabel('User Activity Level', fontsize=12)
    axes[1].set_title('Performance by User Activity (Test)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(activity_levels)
    axes[1].set_ylim(0, max(test_accs) * 1.2)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc, count) in enumerate(zip(bars, test_accs, test_counts)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{acc:.3f}\n({count:,})', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/failure_and_activity_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: failure_and_activity_analysis.pdf")
    
    # Figure 4: Leaderboard Comparison
    test_mrr = test_results['metrics']['mrr']
    
    leaderboard_data = {
        'TPNet': {'mrr': 0.827, 'type': 'GNN'},
        'HyperEvent': {'mrr': 0.810, 'type': 'GNN'},
        'DyGFormer': {'mrr': 0.798, 'type': 'Transformer'},
        'NAT': {'mrr': 0.749, 'type': 'GNN'},
        'Base3': {'mrr': 0.743, 'type': 'Heuristic'},
        'EdgeBank(tw)': {'mrr': 0.571, 'type': 'Heuristic'},
        'EdgeBank(unlimited)': {'mrr': 0.495, 'type': 'Heuristic'},
        'RecencyBank': {'mrr': test_mrr, 'type': 'Heuristic'},
    }
    
    sorted_methods = sorted(leaderboard_data.items(), key=lambda x: x[1]['mrr'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = [m[0] for m in sorted_methods]
    mrrs = [m[1]['mrr'] for m in sorted_methods]
    colors = ['#e74c3c' if m[0] == 'RecencyBank' else 
              '#3498db' if m[1]['type'] == 'GNN' else 
              '#2ecc71' if m[1]['type'] == 'Transformer' else 
              '#f39c12' for m in sorted_methods]
    
    bars = ax.barh(methods, mrrs, color=colors, edgecolor='black', linewidth=1.5)
    
    our_idx = methods.index('RecencyBank')
    bars[our_idx].set_linewidth(3)
    bars[our_idx].set_edgecolor('darkred')
    
    ax.set_xlabel('MRR (Mean Reciprocal Rank)', fontsize=12)
    ax.set_title('tgbl-wiki Leaderboard Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(mrrs) * 1.1)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (method, mrr) in enumerate(zip(methods, mrrs)):
        ax.text(mrr + 0.01, i, f'{mrr:.3f}', va='center', fontsize=10, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Our Method'),
        Patch(facecolor='#3498db', edgecolor='black', label='GNN'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='Transformer'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Heuristic')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/leaderboard_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: leaderboard_comparison.pdf")


def main():
    """Main analysis pipeline."""
    print_section("TGB-WIKI RECENCYBANK ANALYSIS")
    
    # Load and explore
    data_dict = load_and_explore_dataset()
    
    # Create dataset visualizations
    create_visualizations(data_dict)
    
    # Train model
    recency_bank = train_recencybank(data_dict['train_data'])
    
    # Load negative samples
    print_section("Evaluating")
    dataset = data_dict['dataset']
    dataset.load_val_ns()
    dataset.load_test_ns()
    
    val_neg_samples = dataset.ns_sampler.eval_set.get('val')
    test_neg_samples = dataset.ns_sampler.eval_set.get('test')
    
    # Evaluate
    val_results = evaluate_model(
        data_dict['val_data'], val_neg_samples, recency_bank,
        data_dict['out_degree'], data_dict['low_threshold'], 
        data_dict['high_threshold'], "validation"
    )
    
    test_results = evaluate_model(
        data_dict['test_data'], test_neg_samples, recency_bank,
        data_dict['out_degree'], data_dict['low_threshold'], 
        data_dict['high_threshold'], "test"
    )
    
    # Print results
    print(f"\nValidation MRR: {val_results['metrics']['mrr']:.3f}")
    print(f"Test MRR: {test_results['metrics']['mrr']:.3f}")
    
    print(f"\nFailure Analysis:")
    print(f"  Correct: {test_results['failure_stats']['correct']/test_results['matched']*100:.1f}%")
    print(f"  Incorrect: {test_results['failure_stats']['incorrect']/test_results['matched']*100:.1f}%")
    print(f"  Cold start: {test_results['failure_stats']['no_prediction']/test_results['matched']*100:.1f}%")
    
    print(f"\nPerformance by User Activity:")
    for level in ['high', 'medium', 'low']:
        acc = test_results['activity_accuracy'][level]['accuracy']
        count = test_results['activity_accuracy'][level]['count']
        print(f"  {level.capitalize()}: {acc:.3f} accuracy ({count:,} queries)")
    
    # Create analysis visualizations
    create_analysis_visualizations(test_results, val_results)
    
    # Save results
    results = {
        'model': 'RecencyBank',
        'dataset': DATASET_NAME,
        'validation': {
            'metrics': {k: float(v) for k, v in val_results['metrics'].items()},
            'failure_stats': val_results['failure_stats'],
            'activity_accuracy': val_results['activity_accuracy'],
        },
        'test': {
            'metrics': {k: float(v) for k, v in test_results['metrics'].items()},
            'failure_stats': test_results['failure_stats'],
            'activity_accuracy': test_results['activity_accuracy'],
        }
    }
    
    with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print_section("Analysis Complete")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
