"""
TGB-Wiki RecencyBank Analysis - Final Version

Complete pipeline for temporal link prediction with RecencyBank.
Optimized for clarity, correctness, and professional output.

Author: Lorenzo Domenico Attolico
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

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def load_and_explore_dataset():
    """Load tgbl-wiki and perform exploration."""
    print_section("Loading Dataset")
    
    dataset = LinkPropPredDataset(name=DATASET_NAME, root=ROOT_DIR)
    
    num_nodes = dataset.num_nodes
    num_edges = dataset.num_edges
    edge_list = dataset.full_data
    
    sources = edge_list['sources']
    destinations = edge_list['destinations']
    timestamps = edge_list['timestamps']
    
    # Temporal span
    time_span = timestamps[-1] - timestamps[0]
    days = time_span / 86400  # seconds to days
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")
    print(f"  Temporal span: {time_span:,.0f} seconds (~{days:.1f} days)")
    
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
    
    # Compute degrees
    out_degree = Counter(sources)
    in_degree = Counter(destinations)
    
    # Bipartite analysis
    unique_sources = len(set(train_data['sources']))
    unique_dests = len(set(train_data['destinations']))
    
    print(f"\nGraph Structure:")
    print(f"  Unique users (sources): {unique_sources:,}")
    print(f"  Unique pages (destinations): {unique_dests:,}")
    print(f"  Out-degree mean: {np.mean(list(out_degree.values())):.2f}, max: {max(out_degree.values()):,}")
    print(f"  In-degree mean: {np.mean(list(in_degree.values())):.2f}, max: {max(in_degree.values()):,}")
    
    # Activity stratification thresholds
    out_degrees_list = list(out_degree.values())
    low_threshold = np.percentile(out_degrees_list, 33)
    high_threshold = np.percentile(out_degrees_list, 67)
    
    print(f"\nActivity Stratification:")
    print(f"  Low: ≤{low_threshold:.0f} edits (33rd percentile)")
    print(f"  Medium: {low_threshold:.0f}-{high_threshold:.0f} edits")
    print(f"  High: >{high_threshold:.0f} edits (67th percentile)")
    
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
        'time_span_days': days,
    }


def create_visualizations(data_dict):
    """Generate all visualizations."""
    print_section("Generating Visualizations")
    
    out_degree = data_dict['out_degree']
    in_degree = data_dict['in_degree']
    timestamps = data_dict['timestamps']
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    
    # Figure 1: Degree Distribution (NO vertical lines)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    out_degrees = list(out_degree.values())
    axes[0].hist(out_degrees, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].set_xlabel('Out-degree (edits per user)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Out-degree Distribution', fontsize=13, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
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
    """Train RecencyBank."""
    print_section("Training RecencyBank")
    
    recency_bank = RecencyBank()
    
    # Verify if data is already temporally sorted
    ts = train_data['timestamps']
    is_sorted = np.all(ts[:-1] <= ts[1:])
    print(f"Training data temporal ordering: {'already sorted' if is_sorted else 'NOT sorted - will sort'}")
    
    # Sort by timestamp using STABLE sort for deterministic results
    # Mergesort preserves relative order of edges with identical timestamps
    order = np.argsort(train_data['timestamps'], kind='mergesort')
    
    print(f"Processing {len(order):,} edges in chronological order...")
    
    for idx in order:
        src = train_data['sources'][idx]
        dst = train_data['destinations'][idx]
        ts = train_data['timestamps'][idx]
        recency_bank.update(src, dst, ts)
    
    stats = recency_bank.get_statistics()
    print(f"Trained on {stats['num_updates']:,} edges")
    print(f"Tracking {stats['num_sources']:,} unique users")
    print(f"Tracking {stats['num_pairs']:,} unique (user, page) pairs")
    
    with open(f"{OUTPUT_DIR}/recency_bank.pkl", 'wb') as f:
        pickle.dump(recency_bank, f)
    
    return recency_bank


def evaluate_model(data, neg_samples, model, out_degree, low_threshold, high_threshold, split_name="test"):
    """Evaluate with failure analysis and activity stratification."""
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    
    y_pred_pos = []
    y_pred_neg = []
    matched = 0
    
    # For failure analysis with TOP-1 prediction
    correct_count = 0
    incorrect_count = 0
    no_pred_count = 0
    
    activity_predictions = {'low': [], 'medium': [], 'high': []}
    activity_counts = {'low': 0, 'medium': 0, 'high': 0}
    
    print(f"\nEvaluating {len(sources)} queries on {split_name}...")
    
    for i in range(len(sources)):
        src = sources[i]
        dst = destinations[i]
        ts = timestamps[i]
        
        # Get negative samples for this query
        query_key = (src, dst, ts)
        if query_key not in neg_samples:
            continue
        
        neg_dsts = neg_samples[query_key]
        matched += 1
        
        # Score positive edge
        pos_score = model.score_one(src, dst, ts)
        y_pred_pos.append(pos_score)
        
        # Score ALL negative edges
        neg_scores = model.score_batch(
            np.full(len(neg_dsts), src, dtype=np.int64),
            np.array(neg_dsts, dtype=np.int64),
            np.full(len(neg_dsts), ts, dtype=np.int64)
        )
        y_pred_neg.append(neg_scores)
        
        # ===== FAILURE ANALYSIS =====
        # For top-1 prediction: find destination with highest score
        # Check all known destinations for this source
        if src not in model.last_time or len(model.last_time[src]) == 0:
            # Cold start: no history for this source
            no_pred_count += 1
            is_correct = False
        else:
            # Find destination with highest score among known destinations
            best_dst = None
            best_score = -np.inf
            
            for candidate_dst in model.last_time[src].keys():
                score = model.score_one(src, candidate_dst, ts)
                if score > best_score:
                    best_score = score
                    best_dst = candidate_dst
            
            if best_dst == dst:
                correct_count += 1
                is_correct = True
            else:
                incorrect_count += 1
                is_correct = False
        
        # Activity stratification
        user_activity = out_degree.get(src, 0)
        if user_activity <= low_threshold:
            level = 'low'
        elif user_activity <= high_threshold:
            level = 'medium'
        else:
            level = 'high'
        
        activity_predictions[level].append(1.0 if is_correct else 0.0)
        activity_counts[level] += 1
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(sources)} queries...")
    
    print(f"  Matched {matched} queries with negative samples\n")
    
    # Convert to arrays for TGB Evaluator
    y_pred_pos = np.array(y_pred_pos, dtype=np.float64)
    
    # Pad ragged negatives to dense array
    max_len = max(len(negs) for negs in y_pred_neg)
    y_pred_neg_array = np.zeros((len(y_pred_neg), max_len), dtype=np.float64)
    
    for i, negs in enumerate(y_pred_neg):
        y_pred_neg_array[i, :len(negs)] = negs
    
    # Evaluate with TGB official evaluator
    evaluator = Evaluator(name=DATASET_NAME)
    
    input_dict = {
        'y_pred_pos': y_pred_pos,
        'y_pred_neg': y_pred_neg_array,
        'eval_metric': ['mrr', 'hits@']
    }
    
    metrics = evaluator.eval(input_dict)
    
    # Activity accuracy
    activity_acc = {}
    for level in ['low', 'medium', 'high']:
        scores = activity_predictions[level]
        count = activity_counts[level]
        if len(scores) > 0:
            activity_acc[level] = {
                'accuracy': float(np.mean(scores)),
                'count': count,
                'percentage': count / matched * 100
            }
        else:
            activity_acc[level] = {'accuracy': 0.0, 'count': 0, 'percentage': 0.0}
    
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
    """Create failure analysis and leaderboard figures."""
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
    
    # Activity bar chart with counts
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
    test_pcts = [
        test_results['activity_accuracy']['low']['percentage'],
        test_results['activity_accuracy']['medium']['percentage'],
        test_results['activity_accuracy']['high']['percentage']
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
    
    for i, (bar, acc, count, pct) in enumerate(zip(bars, test_accs, test_counts, test_pcts)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{acc:.3f}\n({count:,} queries)\n{pct:.1f}%', 
                     ha='center', va='bottom', fontsize=9)
    
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
    
    # Create visualizations
    create_visualizations(data_dict)
    
    # Train
    recency_bank = train_recencybank(data_dict['train_data'])
    
    # Evaluate
    print_section("Evaluating")
    dataset = data_dict['dataset']
    dataset.load_val_ns()
    dataset.load_test_ns()
    
    val_neg_samples = dataset.ns_sampler.eval_set.get('val')
    test_neg_samples = dataset.ns_sampler.eval_set.get('test')
    
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
    
    print(f"\nFailure Breakdown (Test):")
    print(f"  Correct: {test_results['failure_stats']['correct']/test_results['matched']*100:.1f}%")
    print(f"  Incorrect: {test_results['failure_stats']['incorrect']/test_results['matched']*100:.1f}%")
    print(f"  Cold start: {test_results['failure_stats']['no_prediction']/test_results['matched']*100:.1f}%")
    
    print(f"\nPerformance by User Activity (Test):")
    for level in ['high', 'medium', 'low']:
        acc = test_results['activity_accuracy'][level]['accuracy']
        count = test_results['activity_accuracy'][level]['count']
        pct = test_results['activity_accuracy'][level]['percentage']
        print(f"  {level.capitalize()}: {acc:.3f} accuracy ({count:,} queries, {pct:.1f}%)")
    
    # Create analysis visualizations
    create_analysis_visualizations(test_results, val_results)
    
    # Save results
    results = {
        'model': 'RecencyBank',
        'dataset': {
            'name': DATASET_NAME,
            'nodes': 9227,
            'edges': 157474,
            'time_span_days': data_dict['time_span_days'],
        },
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
