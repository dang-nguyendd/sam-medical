import numpy as np
import pandas as pd
import os
import cv2
from scipy.stats import wilcoxon, friedmanchisquare, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.stats import t
from itertools import combinations

from analysis_stats import (BoundingBox, load_yolo_boxes, load_inference_boxes, calculate_iou,
                            get_ordered_models, get_display_name)
from constants import get_size_categories


class AggregatedObjectSizeAnalyzer:
    """
    Aggregated Object Size Analysis across multiple datasets
    """

    def __init__(self, datasets_info: List[Dict], models: List[str],
                 output_dir: str, iou_threshold: float = 0.5):
        self.datasets_info = datasets_info
        self.models = models
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold

        os.makedirs(output_dir, exist_ok=True)

        # Use configurable size categories from size_config
        self.size_categories = get_size_categories()

        print("\nUsing size categories:")
        for name, (min_val, max_val) in self.size_categories.items():
            print(f"  {name}: {min_val * 100:.4f}% - {max_val * 100:.4f}% of image area")

    def categorize_by_size(self, box: BoundingBox) -> str:
        """Categorize a box by its size"""
        area = box.area()
        for category, (min_area, max_area) in self.size_categories.items():
            if min_area <= area < max_area:
                return category
        return 'large'

    def analyze_single_dataset(self, dataset_info: Dict) -> pd.DataFrame:
        """Analyze object size performance for a single dataset"""
        dataset_name = dataset_info['name']
        gt_folder = dataset_info['gt_folder']
        inference_root = dataset_info['inference_root']
        image_folder = dataset_info['image_folder']

        print(f"  Analyzing dataset: {dataset_name}")

        # Get all image files
        image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                       if f.endswith('.txt')]

        # First pass: Count ground truth distribution across size categories
        print(f"\n  {'=' * 70}")
        print(f"  GROUND TRUTH SIZE DISTRIBUTION - {dataset_name}")
        print(f"  {'=' * 70}")

        gt_distribution = {cat: 0 for cat in self.size_categories.keys()}
        total_gt_boxes = 0

        for img_name in image_files:
            gt_path = os.path.join(gt_folder, f"{img_name}.txt")

            # Get image dimensions
            img_height, img_width = None, None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                image_path = os.path.join(image_folder, f"{img_name}{ext}")
                if os.path.exists(image_path):
                    import cv2
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        break

            if img_width is None and img_height is None:
                img_width, img_height = 640, 640

            # Load and categorize ground truth boxes
            gt_boxes = load_yolo_boxes(gt_path)
            for gt_box in gt_boxes:
                cat = self.categorize_by_size(gt_box)
                gt_distribution[cat] += 1
                total_gt_boxes += 1

        # Print distribution
        print(f"  Total ground truth boxes: {total_gt_boxes}")
        print(f"  {'-' * 70}")
        for cat in self.size_categories.keys():
            count = gt_distribution[cat]
            pct = (count / total_gt_boxes * 100) if total_gt_boxes > 0 else 0
            min_area, max_area = self.size_categories[cat]
            print(
                f"  {cat:10s}: {count:6d} boxes ({pct:5.1f}%) | Area range: {min_area * 100:.4f}% - {max_area * 100:.4f}%")
        print(f"  {'=' * 70}\n")

        all_results = []

        for model in self.models:
            # Initialize counters for each size category
            category_stats = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0}
                              for cat in self.size_categories.keys()}

            for img_name in image_files:
                gt_path = os.path.join(gt_folder, f"{img_name}.txt")
                inf_path = os.path.join(inference_root, model, f"{img_name}.txt")

                img_height, img_width = None, None
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    image_path = os.path.join(image_folder, f"{img_name}{ext}")
                    if os.path.exists(image_path):
                        import cv2
                        img = cv2.imread(image_path)
                        if img is not None:
                            img_height, img_width = img.shape[:2]
                            break

                if img_width is None and img_height is None:
                    img_width, img_height = 640, 640

                gt_boxes = load_yolo_boxes(gt_path)
                pred_boxes = load_inference_boxes(inf_path, img_width, img_height)

                # Categorize ground truth boxes
                gt_by_category = {cat: [] for cat in self.size_categories.keys()}
                for gt_box in gt_boxes:
                    cat = self.categorize_by_size(gt_box)
                    gt_by_category[cat].append(gt_box)
                    category_stats[cat]['total_gt'] += 1

                # Match predictions to ground truth
                matched_gt = set()
                matched_pred = set()

                for i, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_gt_idx = -1
                    best_category = None

                    for cat, gt_list in gt_by_category.items():
                        for j, gt_box in enumerate(gt_list):
                            if gt_box.cls != pred_box.cls:
                                continue

                            # Create unique identifier for gt_box
                            gt_global_idx = (cat, j)
                            if gt_global_idx in matched_gt:
                                continue

                            iou = calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                                best_category = cat

                    if best_iou >= self.iou_threshold and best_category is not None:
                        # True positive
                        category_stats[best_category]['tp'] += 1
                        matched_gt.add((best_category, best_gt_idx))
                        matched_pred.add(i)
                    else:
                        # False positive - categorize by prediction size
                        pred_cat = self.categorize_by_size(pred_box)
                        category_stats[pred_cat]['fp'] += 1

                # Count false negatives (unmatched GT)
                for cat, gt_list in gt_by_category.items():
                    for j, gt_box in enumerate(gt_list):
                        if (cat, j) not in matched_gt:
                            category_stats[cat]['fn'] += 1

            # Calculate metrics for each category
            for cat, stats_dict in category_stats.items():
                tp = stats_dict['tp']
                fp = stats_dict['fp']
                fn = stats_dict['fn']

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                all_results.append({
                    'dataset': dataset_name,
                    'model': model,
                    'size_category': cat,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'total_gt': stats_dict['total_gt'],
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        return pd.DataFrame(all_results)

    def analyze_all_datasets(self) -> pd.DataFrame:
        """Analyze all datasets and aggregate results"""
        all_dfs = []

        # Store GT distributions for final summary
        all_gt_distributions = {}

        for dataset_info in self.datasets_info:
            df = self.analyze_single_dataset(dataset_info)
            all_dfs.append(df)

            # Extract GT distribution from results
            dataset_name = dataset_info['name']
            gt_dist = df.groupby('size_category')['total_gt'].first().to_dict()
            all_gt_distributions[dataset_name] = gt_dist

        # Combine all results
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Save combined results
        output_path = os.path.join(self.output_dir, 'size_category_all_datasets.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\nAggregated size category results saved to: {output_path}")

        # Also save aggregated summary (summed across datasets)
        aggregated_df = self._aggregate_across_datasets(combined_df)
        agg_output_path = os.path.join(self.output_dir, 'size_category_aggregated_summary.csv')
        aggregated_df.to_csv(agg_output_path, index=False)
        print(f"Aggregated summary saved to: {agg_output_path}")

        # Print comprehensive summary
        print(f"\n{'=' * 80}")
        print("GROUND TRUTH DISTRIBUTION SUMMARY - ALL DATASETS")
        print(f"{'=' * 80}")

        # Aggregate across all datasets
        total_gt_all_datasets = {cat: 0 for cat in self.size_categories.keys()}
        overall_total = 0

        for dataset_name, gt_dist in all_gt_distributions.items():
            dataset_total = sum(gt_dist.values())
            overall_total += dataset_total

            print(f"\n{dataset_name}:")
            print(f"  {'─' * 76}")
            print(f"  {'Category':<12} {'Count':>8} {'Percentage':>12} {'Area Range':<40}")
            print(f"  {'─' * 76}")

            for cat in self.size_categories.keys():
                count = gt_dist.get(cat, 0)
                pct = (count / dataset_total * 100) if dataset_total > 0 else 0
                min_area, max_area = self.size_categories[cat]
                area_str = f"{min_area * 100:.4f}% - {max_area * 100:.4f}%"
                print(f"  {cat:<12} {count:>8} {pct:>11.1f}% {area_str:<40}")
                total_gt_all_datasets[cat] += count

            print(f"  {'─' * 76}")
            print(f"  {'TOTAL':<12} {dataset_total:>8}")

        # Overall summary
        print(f"\n{'=' * 80}")
        print("AGGREGATED ACROSS ALL DATASETS:")
        print(f"{'=' * 80}")
        print(f"  {'Category':<12} {'Count':>8} {'Percentage':>12} {'Area Range':<40}")
        print(f"  {'─' * 76}")

        for cat in self.size_categories.keys():
            count = total_gt_all_datasets[cat]
            pct = (count / overall_total * 100) if overall_total > 0 else 0
            min_area, max_area = self.size_categories[cat]
            area_str = f"{min_area * 100:.4f}% - {max_area * 100:.4f}%"
            print(f"  {cat:<12} {count:>8} {pct:>11.1f}% {area_str:<40}")

        print(f"  {'─' * 76}")
        print(f"  {'TOTAL':<12} {overall_total:>8}")
        print(f"{'=' * 80}\n")

        # Save GT distribution summary to file
        gt_summary_path = os.path.join(self.output_dir, 'ground_truth_size_distribution.txt')
        with open(gt_summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GROUND TRUTH SIZE DISTRIBUTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("SIZE CATEGORIES DEFINITION:\n")
            f.write("-" * 80 + "\n")
            for cat, (min_area, max_area) in self.size_categories.items():
                f.write(f"  {cat:<12}: {min_area * 100:>8.4f}% - {max_area * 100:>8.4f}% of image area\n")
            f.write("\n")

            # Per-dataset distributions
            for dataset_name, gt_dist in all_gt_distributions.items():
                dataset_total = sum(gt_dist.values())

                f.write(f"\n{dataset_name}:\n")
                f.write("  " + "─" * 76 + "\n")
                f.write(f"  {'Category':<12} {'Count':>8} {'Percentage':>12} {'Area Range':<40}\n")
                f.write("  " + "─" * 76 + "\n")

                for cat in self.size_categories.keys():
                    count = gt_dist.get(cat, 0)
                    pct = (count / dataset_total * 100) if dataset_total > 0 else 0
                    min_area, max_area = self.size_categories[cat]
                    area_str = f"{min_area * 100:.4f}% - {max_area * 100:.4f}%"
                    f.write(f"  {cat:<12} {count:>8} {pct:>11.1f}% {area_str:<40}\n")

                f.write("  " + "─" * 76 + "\n")
                f.write(f"  {'TOTAL':<12} {dataset_total:>8}\n")

            # Overall summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("AGGREGATED ACROSS ALL DATASETS:\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {'Category':<12} {'Count':>8} {'Percentage':>12} {'Area Range':<40}\n")
            f.write("  " + "─" * 76 + "\n")

            for cat in self.size_categories.keys():
                count = total_gt_all_datasets[cat]
                pct = (count / overall_total * 100) if overall_total > 0 else 0
                min_area, max_area = self.size_categories[cat]
                area_str = f"{min_area * 100:.4f}% - {max_area * 100:.4f}%"
                f.write(f"  {cat:<12} {count:>8} {pct:>11.1f}% {area_str:<40}\n")

            f.write("  " + "─" * 76 + "\n")
            f.write(f"  {'TOTAL':<12} {overall_total:>8}\n")
            f.write("=" * 80 + "\n")

        print(f"Ground truth distribution summary saved to: {gt_summary_path}\n")

        return combined_df

    def _aggregate_across_datasets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics across all datasets"""
        # Group by model and size_category, sum TP/FP/FN
        grouped = df.groupby(['model', 'size_category']).agg({
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'total_gt': 'sum'
        }).reset_index()

        # Recalculate metrics
        grouped['precision'] = grouped.apply(
            lambda row: row['tp'] / (row['tp'] + row['fp']) if (row['tp'] + row['fp']) > 0 else 0,
            axis=1
        )
        grouped['recall'] = grouped.apply(
            lambda row: row['tp'] / (row['tp'] + row['fn']) if (row['tp'] + row['fn']) > 0 else 0,
            axis=1
        )
        grouped['f1'] = grouped.apply(
            lambda row: 2 * row['precision'] * row['recall'] / (row['precision'] + row['recall'])
            if (row['precision'] + row['recall']) > 0 else 0,
            axis=1
        )

        return grouped

    def plot_aggregated_results(self, df: pd.DataFrame):
        """Create visualizations for aggregated results"""
        # Aggregate across datasets for plotting
        agg_df = self._aggregate_across_datasets(df)

        # Order models and add display names
        agg_df = agg_df.copy()
        ordered_models = get_ordered_models(agg_df['model'].unique())
        agg_df['model'] = pd.Categorical(agg_df['model'], categories=ordered_models, ordered=True)
        agg_df = agg_df.sort_values('model')
        agg_df['display_name'] = agg_df['model'].apply(get_display_name)

        # Extract GT distribution
        gt_dist_df = df.groupby('size_category')['total_gt'].sum().reset_index()
        gt_dist_df.columns = ['size_category', 'gt_count']

        # NEW: Create comprehensive visualization with GT distribution
        categories = list(self.size_categories.keys())

        fig = plt.figure(figsize=(16, 10))

        # Plot 1: Ground Truth Distribution
        ax1 = plt.subplot(2, 2, 1)
        gt_counts = [gt_dist_df[gt_dist_df['size_category'] == cat]['gt_count'].values[0]
                     if len(gt_dist_df[gt_dist_df['size_category'] == cat]) > 0 else 0
                     for cat in categories]

        colors_gt = plt.cm.viridis(np.linspace(0, 0.8, len(categories)))
        bars = ax1.bar(categories, gt_counts, color=colors_gt, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Size Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of GT Boxes', fontsize=12, fontweight='bold')
        ax1.set_title('Ground Truth Distribution by Size', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Plot 2: Recall by Size Category (all models)
        ax2 = plt.subplot(2, 2, 2)
        pivot_recall = agg_df.pivot(index='display_name', columns='size_category', values='recall')
        pivot_recall = pivot_recall[[c for c in categories if c in pivot_recall.columns]]  # Ensure correct order

        x = np.arange(len(categories))
        width = 0.08
        multiplier = 0

        for model in pivot_recall.index:
            offset = width * multiplier
            values = pivot_recall.loc[model].values
            ax2.bar(x + offset, values, width, label=model, edgecolor='black', linewidth=0.5)
            multiplier += 1

        ax2.set_xlabel('Size Category', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_title('Recall by Size Category', fontsize=13, fontweight='bold')
        ax2.set_xticks(x + width * (len(pivot_recall.index) - 1) / 2)
        ax2.set_xticklabels(categories)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1])

        # Plot 3: F1 Score by Size Category
        ax3 = plt.subplot(2, 2, 3)
        pivot_f1 = agg_df.pivot(index='display_name', columns='size_category', values='f1')
        pivot_f1 = pivot_f1[[c for c in categories if c in pivot_f1.columns]]  # Ensure correct order

        x = np.arange(len(categories))
        multiplier = 0

        for model in pivot_f1.index:
            offset = width * multiplier
            values = pivot_f1.loc[model].values
            ax3.bar(x + offset, values, width, label=model, edgecolor='black', linewidth=0.5)
            multiplier += 1

        ax3.set_xlabel('Size Category', fontsize=12, fontweight='bold')
        ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax3.set_title('F1 Score by Size Category', fontsize=13, fontweight='bold')
        ax3.set_xticks(x + width * (len(pivot_f1.index) - 1) / 2)
        ax3.set_xticklabels(categories)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1])

        # Plot 4: Detection Rate vs GT Distribution
        ax4 = plt.subplot(2, 2, 4)

        # Calculate average recall per category
        avg_recall = agg_df.groupby('size_category')['recall'].mean().reset_index()
        avg_recall = avg_recall.set_index('size_category').loc[categories].reset_index()

        # Normalize GT counts for comparison
        total_gt = sum(gt_counts)
        gt_pcts = [count / total_gt * 100 if total_gt > 0 else 0 for count in gt_counts]
        recall_pcts = avg_recall['recall'].values * 100

        x_pos = np.arange(len(categories))
        width = 0.35

        bars1 = ax4.bar(x_pos - width / 2, gt_pcts, width, label='GT Distribution %',
                        color='steelblue', edgecolor='black', linewidth=1.5)
        bars2 = ax4.bar(x_pos + width / 2, recall_pcts, width, label='Avg Recall %',
                        color='coral', edgecolor='black', linewidth=1.5)

        ax4.set_xlabel('Size Category', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax4.set_title('GT Distribution vs Average Detection Rate', fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('Object Size Category Analysis - Aggregated Results',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 0.95, 0.99])
        plt.savefig(os.path.join(self.output_dir, 'size_category_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Original Plot 1: F1 score by size category (aggregated)
        plt.figure(figsize=(12, 6))

        categories = list(self.size_categories.keys())
        x = np.arange(len(categories))
        width = 0.1

        for i, model in enumerate(self.models):
            model_data = agg_df[agg_df['model'] == model]
            f1_scores = [model_data[model_data['size_category'] == cat]['f1'].values[0]
                         if len(model_data[model_data['size_category'] == cat]) > 0 else 0
                         for cat in categories]

            plt.bar(x + i * width, f1_scores, width, label=model.upper())

        plt.xlabel('Object Size Category', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score by Object Size (All Datasets)', fontsize=14, fontweight='bold')
        EN_DASH = '\u2013'
        categories_plot = ['small\n<0.5%', f'medium\n0.5% {EN_DASH} 1%', f'large\n1% {EN_DASH} 5%', 'very large\n>5%']
        plt.xticks(x + width * (len(self.models) - 1) / 2, categories_plot)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_size_aggregated.png'), dpi=300)
        plt.close()

        # Plot 2: Heatmap of F1 scores (aggregated)
        pivot_df = agg_df.pivot(index='model', columns='size_category', values='f1')
        pivot_df = pivot_df[categories]  # Ensure correct order

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'F1 Score'})
        plt.title('F1 Score by Size Category', fontsize=14, fontweight='bold')
        plt.xlabel('Object Size Category', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_heatmap_aggregated.png'), dpi=300)
        plt.close()

        # Plot 3: Per-dataset comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        datasets = df['dataset'].unique()
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break

            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]

            for model in self.models:
                model_data = dataset_df[dataset_df['model'] == model]
                f1_scores = [model_data[model_data['size_category'] == cat]['f1'].values[0]
                             if len(model_data[model_data['size_category'] == cat]) > 0 else 0
                             for cat in categories]

                EN_DASH = '\u2013'
                categories_plot = ['small\n<0.5%', f'medium\n0.5% {EN_DASH} 1%', f'large\n1% {EN_DASH} 5%', 'very large\n>5%']
                ax.plot(categories_plot, f1_scores, marker='o', label=model.upper(), linewidth=2)

            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Size Category', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('F1 Score by Size Category - Per Dataset Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_by_size_per_dataset.png'), dpi=300)
        plt.close()

        print(f"Aggregated size analysis plots saved to: {self.output_dir}")


class AggregatedStatisticalAnalyzer:
    """
    Comprehensive Statistical Analysis across multiple datasets using:
    - Friedman test (non-parametric repeated measures)
    - Nemenyi post-hoc test
    - Wilcoxon signed-rank test with Bonferroni correction
    - Effect size calculations (Cohen's d, rank biserial correlation)
    - Critical difference diagrams
    """

    def __init__(self, datasets_info: List[Dict], output_dir: str):
        self.datasets_info = datasets_info
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_all_results(self) -> pd.DataFrame:
        """Load and combine results from all datasets"""
        all_dfs = []

        for dataset_info in self.datasets_info:
            csv_path = dataset_info['results_csv']
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['dataset'] = dataset_info['name']
                all_dfs.append(df)

        if len(all_dfs) == 0:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    def compute_aggregated_confidence_intervals(self, metric: str = 'box_mAP@50',
                                                confidence: float = 0.95) -> pd.DataFrame:
        """Compute confidence intervals aggregated across datasets"""
        combined_df = self.load_all_results()

        if len(combined_df) == 0:
            return pd.DataFrame()

        results = []

        for model in combined_df['model'].unique():
            model_data = combined_df[combined_df['model'] == model][metric].values

            if len(model_data) == 0:
                continue

            mean_val = np.mean(model_data)
            std_val = np.std(model_data, ddof=1)
            median_val = np.median(model_data)
            n = len(model_data)

            # Calculate confidence interval
            confidence_level = confidence
            df_val = n - 1
            t_crit = t.ppf((1 + confidence_level) / 2, df_val)
            margin_error = t_crit * (std_val / np.sqrt(n))

            results.append({
                'model': model,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'n_datasets': n,
                'ci_lower': mean_val - margin_error,
                'ci_upper': mean_val + margin_error,
                'confidence_level': confidence_level,
                'sem': std_val / np.sqrt(n)  # Standard error of mean
            })

        df = pd.DataFrame(results)

        output_path = os.path.join(self.output_dir, f'{metric}_confidence_intervals_aggregated.csv')
        df.to_csv(output_path, index=False)
        print(f"Confidence intervals saved to: {output_path}")

        return df

    def plot_aggregated_confidence_intervals(self, df: pd.DataFrame, metric: str):
        """Plot confidence intervals across models"""
        if len(df) == 0:
            return

        plt.figure(figsize=(14, 6))

        # Order models and add display names
        df = df.copy()
        ordered_models = get_ordered_models(df['model'].unique())
        df['model'] = pd.Categorical(df['model'], categories=ordered_models, ordered=True)
        df = df.sort_values('model')
        df['display_name'] = df['model'].apply(get_display_name)

        models = df['display_name'].values
        means = df['mean'].values
        ci_lower = df['ci_lower'].values
        ci_upper = df['ci_upper'].values

        x = np.arange(len(models))

        plt.errorbar(x, means, yerr=[means - ci_lower, ci_upper - means],
                     fmt='o', capsize=5, capthick=2, markersize=10,
                     linewidth=2, color='steelblue', elinewidth=2)

        # Add value labels
        for i, (model, mean) in enumerate(zip(models, means)):
            plt.text(i, mean, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel(f'{metric} (Mean ± 95% CI)', fontsize=12, fontweight='bold')
        plt.title(
            f'Model Performance with 95% Confidence Intervals\n(Aggregated across {df["n_datasets"].iloc[0]:.0f} datasets)',
            fontsize=14, fontweight='bold')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric}_ci_plot_aggregated.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def friedman_test(self, metric: str = 'box_mAP@50') -> Dict:
        """
        Perform Friedman test to check if there are significant differences between models.

        The Friedman test is a non-parametric test for repeated measures (same datasets tested
        across all models). It's the non-parametric equivalent of repeated measures ANOVA.

        H0: All models have the same performance
        H1: At least one model performs differently
        """
        combined_df = self.load_all_results()

        if len(combined_df) == 0:
            return {}

        # Prepare data: pivot to have datasets as rows, models as columns
        pivot_df = combined_df.pivot(index='dataset', columns='model', values=metric)

        # Remove any rows with missing values
        pivot_df = pivot_df.dropna()

        if len(pivot_df) < 2:
            print(f"Warning: Need at least 2 datasets for Friedman test. Found {len(pivot_df)}")
            return {}

        # Extract data for each model
        model_data = [pivot_df[model].values for model in pivot_df.columns]

        # Perform Friedman test
        if len(model_data) < 3:
            print(f"Warning: Friedman test requires at least 3 models. Found {len(model_data)}")
            return {}

        try:
            statistic, p_value = friedmanchisquare(*model_data)
        except Exception as e:
            print(f"Error performing Friedman test: {e}")
            return {}

        # Calculate effect size (Kendall's W)
        n = len(pivot_df)  # number of datasets
        k = len(model_data)  # number of models
        kendalls_w = statistic / (n * (k - 1))

        result = {
            'test': 'Friedman',
            'statistic': statistic,
            'p_value': p_value,
            'n_datasets': n,
            'n_models': k,
            'kendalls_w': kendalls_w,
            'significant': p_value < 0.05,
            'models': list(pivot_df.columns)
        }

        # Save results
        result_df = pd.DataFrame([result])
        output_path = os.path.join(self.output_dir, f'{metric}_friedman_test.csv')
        result_df.to_csv(output_path, index=False)
        print(f"\nFriedman Test Results:")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4e}")
        print(f"  Kendall's W (effect size): {kendalls_w:.4f}")
        print(f"  Significant: {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p ≥ 0.05)'}")
        print(f"  Results saved to: {output_path}")

        return result

    def nemenyi_post_hoc(self, metric: str = 'box_mAP@50', alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform Nemenyi post-hoc test for pairwise comparisons after Friedman test.

        The Nemenyi test computes critical differences between average ranks of models.
        Models are significantly different if their rank difference exceeds the critical difference.
        """
        combined_df = self.load_all_results()

        if len(combined_df) == 0:
            return pd.DataFrame()

        # Prepare data
        pivot_df = combined_df.pivot(index='dataset', columns='model', values=metric)
        pivot_df = pivot_df.dropna()

        n = len(pivot_df)  # number of datasets
        k = len(pivot_df.columns)  # number of models

        # Compute ranks for each dataset (across models)
        rank_df = pivot_df.rank(axis=1, ascending=False)

        # Compute average ranks for each model
        avg_ranks = rank_df.mean(axis=0)

        # Critical difference for Nemenyi test
        # CD = q_alpha * sqrt(k(k+1) / (6n))
        # q_alpha values for different significance levels (from Nemenyi table)
        q_alpha_values = {
            0.05: {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
            0.10: {3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}
        }

        if k in q_alpha_values.get(alpha, {}):
            q_alpha = q_alpha_values[alpha][k]
        else:
            # Approximation for larger k
            from scipy.stats import studentized_range
            q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)

        critical_difference = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        # Pairwise comparisons
        results = []
        models = list(pivot_df.columns)

        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i >= j:
                    continue

                rank_diff = abs(avg_ranks[model_a] - avg_ranks[model_b])
                is_significant = rank_diff > critical_difference

                # Determine which model is better (lower rank is better)
                better_model = model_a if avg_ranks[model_a] < avg_ranks[model_b] else model_b
                mean_a = pivot_df[model_a].mean()
                mean_b = pivot_df[model_b].mean()

                results.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'avg_rank_a': avg_ranks[model_a],
                    'avg_rank_b': avg_ranks[model_b],
                    'rank_difference': rank_diff,
                    'critical_difference': critical_difference,
                    'significant': is_significant,
                    'better_model': better_model if is_significant else 'ns',
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'mean_difference': abs(mean_a - mean_b)
                })

        df = pd.DataFrame(results)

        # Save results
        output_path = os.path.join(self.output_dir, f'{metric}_nemenyi_post_hoc.csv')
        df.to_csv(output_path, index=False)

        # Save average ranks
        rank_summary = pd.DataFrame({
            'model': avg_ranks.index,
            'average_rank': avg_ranks.values,
            'mean_performance': [pivot_df[model].mean() for model in avg_ranks.index]
        }).sort_values('average_rank')

        rank_path = os.path.join(self.output_dir, f'{metric}_average_ranks.csv')
        rank_summary.to_csv(rank_path, index=False)

        print(f"\nNemenyi Post-Hoc Test Results:")
        print(f"  Critical Difference: {critical_difference:.4f}")
        print(f"  Significant comparisons: {df['significant'].sum()} out of {len(df)}")
        print(f"  Results saved to: {output_path}")
        print(f"  Average ranks saved to: {rank_path}")

        return df

    def wilcoxon_post_hoc_bonferroni(self, metric: str = 'box_mAP@50', alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform pairwise Wilcoxon signed-rank tests with Bonferroni correction.

        This is more conservative than Nemenyi but provides exact p-values for each comparison.
        """
        combined_df = self.load_all_results()

        if len(combined_df) == 0:
            return pd.DataFrame()

        # Prepare data
        pivot_df = combined_df.pivot(index='dataset', columns='model', values=metric)
        pivot_df = pivot_df.dropna()

        models = list(pivot_df.columns)
        n_comparisons = len(list(combinations(models, 2)))
        bonferroni_alpha = alpha / n_comparisons

        results = []

        for model_a, model_b in combinations(models, 2):
            data_a = pivot_df[model_a].values
            data_b = pivot_df[model_b].values

            # Wilcoxon signed-rank test
            try:
                statistic, p_value = wilcoxon(data_a, data_b, alternative='two-sided')
            except Exception as e:
                print(f"Warning: Wilcoxon test failed for {model_a} vs {model_b}: {e}")
                p_value = 1.0
                statistic = 0

            # Effect size: rank-biserial correlation
            n = len(data_a)
            r = 1 - (2 * statistic) / (n * (n + 1))

            # Cohen's d
            mean_diff = np.mean(data_a - data_b)
            std_diff = np.std(data_a - data_b, ddof=1)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0

            is_significant = p_value < bonferroni_alpha
            better_model = model_a if np.mean(data_a) > np.mean(data_b) else model_b

            results.append({
                'model_a': model_a,
                'model_b': model_b,
                'mean_a': np.mean(data_a),
                'mean_b': np.mean(data_b),
                'mean_difference': mean_diff,
                'p_value': p_value,
                'p_value_bonferroni': p_value * n_comparisons,  # adjusted p-value
                'bonferroni_alpha': bonferroni_alpha,
                'significant': is_significant,
                'cohens_d': cohens_d,
                'rank_biserial_r': r,
                'better_model': better_model if is_significant else 'ns'
            })

        df = pd.DataFrame(results)

        # Add significance symbols
        df['significance'] = df['p_value_bonferroni'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )

        # Save results
        output_path = os.path.join(self.output_dir, f'{metric}_wilcoxon_bonferroni.csv')
        df.to_csv(output_path, index=False)

        print(f"\nWilcoxon Signed-Rank Test (Bonferroni corrected):")
        print(f"  Number of comparisons: {n_comparisons}")
        print(f"  Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
        print(f"  Significant comparisons: {df['significant'].sum()} out of {len(df)}")
        print(f"  Results saved to: {output_path}")

        return df

    def compute_effect_sizes(self, metric: str = 'box_mAP@50') -> pd.DataFrame:
        """
        Compute various effect size measures for all pairwise comparisons.
        """
        combined_df = self.load_all_results()

        if len(combined_df) == 0:
            return pd.DataFrame()

        pivot_df = combined_df.pivot(index='dataset', columns='model', values=metric)
        pivot_df = pivot_df.dropna()

        models = list(pivot_df.columns)
        results = []

        for model_a, model_b in combinations(models, 2):
            data_a = pivot_df[model_a].values
            data_b = pivot_df[model_b].values

            # Cohen's d
            mean_diff = np.mean(data_a) - np.mean(data_b)
            pooled_std = np.sqrt((np.var(data_a, ddof=1) + np.var(data_b, ddof=1)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            # Hedge's g (corrected for small sample size)
            n = len(data_a)
            correction = (n - 3) / (n - 2.25) * np.sqrt((n - 2) / n)
            hedges_g = cohens_d * correction

            # Probability of superiority (common language effect size)
            superior_count = sum([a > b for a, b in zip(data_a, data_b)])
            prob_superiority = superior_count / n

            # Interpret effect size
            def interpret_d(d):
                abs_d = abs(d)
                if abs_d < 0.2:
                    return 'negligible'
                elif abs_d < 0.5:
                    return 'small'
                elif abs_d < 0.8:
                    return 'medium'
                else:
                    return 'large'

            results.append({
                'model_a': model_a,
                'model_b': model_b,
                'mean_a': np.mean(data_a),
                'mean_b': np.mean(data_b),
                'mean_difference': mean_diff,
                'cohens_d': cohens_d,
                'hedges_g': hedges_g,
                'effect_size_interpretation': interpret_d(cohens_d),
                'probability_superiority': prob_superiority,
                'better_model': model_a if mean_diff > 0 else model_b
            })

        df = pd.DataFrame(results)

        output_path = os.path.join(self.output_dir, f'{metric}_effect_sizes.csv')
        df.to_csv(output_path, index=False)
        print(f"\nEffect sizes saved to: {output_path}")

        return df

    def create_critical_difference_diagram(self, metric: str = 'box_mAP@50', alpha: float = 0.05):
        """
        Create a Critical Difference (CD) diagram showing which models are significantly different.

        Models connected by a horizontal line are NOT significantly different.
        """
        combined_df = self.load_all_results()

        if len(combined_df) == 0:
            return

        # Get average ranks
        pivot_df = combined_df.pivot(index='dataset', columns='model', values=metric)
        pivot_df = pivot_df.dropna()

        rank_df = pivot_df.rank(axis=1, ascending=False)
        avg_ranks = rank_df.mean(axis=0).sort_values()

        # Get Nemenyi results
        nemenyi_df = self.nemenyi_post_hoc(metric=metric, alpha=alpha)

        if len(nemenyi_df) == 0:
            return

        critical_difference = nemenyi_df['critical_difference'].iloc[0]

        # Create the diagram
        fig, ax = plt.subplots(figsize=(14, 8))

        models = avg_ranks.index.tolist()
        ranks = avg_ranks.values
        n_models = len(models)

        # Plot models
        y_positions = np.arange(n_models)
        ax.barh(y_positions, ranks, color='steelblue', alpha=0.7, edgecolor='black')

        # Add rank values
        for i, (model, rank) in enumerate(zip(models, ranks)):
            ax.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontweight='bold')

        ax.set_yticks(y_positions)
        ax.set_yticklabels([m.upper() for m in models], fontsize=11)
        ax.set_xlabel('Average Rank', fontsize=12, fontweight='bold')
        ax.set_title(f'Critical Difference Diagram - {metric}\n(CD = {critical_difference:.3f})', fontsize=14,
                     fontweight='bold')

        # Draw lines connecting non-significant groups
        # Find groups of models that are not significantly different
        groups = []
        for i in range(n_models):
            group = [i]
            for j in range(i + 1, n_models):
                if abs(ranks[j] - ranks[i]) <= critical_difference:
                    group.append(j)
                else:
                    break
            if len(group) > 1:
                groups.append(group)

        # Draw connecting lines for non-significant groups
        offset = -0.3
        for group_idx, group in enumerate(groups):
            if len(group) > 1:
                y_min = min(group)
                y_max = max(group)
                x_pos = max(ranks[group]) + 0.5 + offset * group_idx

                # Draw line
                ax.plot([x_pos, x_pos], [y_min, y_max], color='red', linewidth=3, alpha=0.6)

                # Add horizontal bars at ends
                bar_length = 0.15
                ax.plot([x_pos - bar_length, x_pos + bar_length], [y_min, y_min], color='red', linewidth=3, alpha=0.6)
                ax.plot([x_pos - bar_length, x_pos + bar_length], [y_max, y_max], color='red', linewidth=3, alpha=0.6)

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(ranks) + 2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric}_critical_difference_diagram.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Critical difference diagram saved to: {self.output_dir}")

    def create_aggregated_significance_heatmap(self, metric: str = 'box_mAP@50'):
        """
        Create comprehensive significance heatmap using Wilcoxon + Bonferroni results.
        """
        df = self.wilcoxon_post_hoc_bonferroni(metric=metric)

        if len(df) == 0:
            return

        models = get_ordered_models(list(set(df['model_a'].unique()) | set(df['model_b'].unique())))
        display_names = [get_display_name(m) for m in models]

        # Create matrix for p-values and performance differences
        p_matrix = np.ones((len(models), len(models)))
        perf_matrix = np.zeros((len(models), len(models)))

        for _, row in df.iterrows():
            i = models.index(row['model_a'])
            j = models.index(row['model_b'])
            p_matrix[i, j] = row['p_value_bonferroni']
            p_matrix[j, i] = row['p_value_bonferroni']
            perf_matrix[i, j] = row['mean_a'] - row['mean_b']
            perf_matrix[j, i] = row['mean_b'] - row['mean_a']

        # Create annotation matrix combining p-values and significance
        annot_matrix = np.empty((len(models), len(models)), dtype=object)
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    annot_matrix[i, j] = '—'
                else:
                    p = p_matrix[i, j]
                    if p < 0.001:
                        sig = '***'
                    elif p < 0.01:
                        sig = '**'
                    elif p < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    annot_matrix[i, j] = f'{p:.3f}\n{sig}'

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Heatmap 1: P-values
        sns.heatmap(p_matrix, annot=annot_matrix, fmt='',
                    cmap='RdYlGn', vmin=0, vmax=0.1,
                    xticklabels=display_names,
                    yticklabels=display_names,
                    cbar_kws={'label': 'Adjusted p-value (Bonferroni)'},
                    ax=ax1, linewidths=0.5, linecolor='gray')
        ax1.set_title('Statistical Significance (Bonferroni-corrected)\n*** p<0.001, ** p<0.01, * p<0.05', fontsize=12,
                      fontweight='bold')

        # Heatmap 2: Performance differences
        sns.heatmap(perf_matrix, annot=True, fmt='.3f',
                    cmap='RdBu_r', center=0,
                    xticklabels=display_names,
                    yticklabels=display_names,
                    cbar_kws={'label': f'{metric} Difference (Row - Column)'},
                    ax=ax2, linewidths=0.5, linecolor='gray')
        ax2.set_title(f'Performance Differences\n(Positive = Row model better)', fontsize=12, fontweight='bold')

        plt.suptitle(f'Comprehensive Statistical Comparison - {metric}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric}_significance_heatmap_comprehensive.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        print(f"Comprehensive significance heatmap saved to: {self.output_dir}")

    def generate_comprehensive_report(self, metrics: List[str] = None):
        """
        Generate a comprehensive statistical analysis report for all metrics.
        """
        if metrics is None:
            metrics = ['box_mAP@50', 'box_mAP@50-95', 'box_mean_f1',
                       'box_mean_precision', 'box_mean_recall']

        report_path = os.path.join(self.output_dir, 'comprehensive_statistical_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            combined_df = self.load_all_results()

            if len(combined_df) == 0:
                f.write("No data available for analysis.\n")
                return

            f.write(f"Number of datasets: {combined_df['dataset'].nunique()}\n")
            f.write(f"Datasets: {', '.join(combined_df['dataset'].unique())}\n")
            f.write(f"Number of models: {combined_df['model'].nunique()}\n")
            f.write(f"Models: {', '.join(combined_df['model'].unique())}\n\n")

            for metric in metrics:
                if metric not in combined_df.columns:
                    continue

                f.write("\n" + "=" * 80 + "\n")
                f.write(f"METRIC: {metric}\n")
                f.write("=" * 80 + "\n\n")

                # Descriptive statistics
                f.write("1. DESCRIPTIVE STATISTICS\n")
                f.write("-" * 40 + "\n")

                pivot_df = combined_df.pivot(index='dataset', columns='model', values=metric)
                pivot_df = pivot_df.dropna()

                for model in pivot_df.columns:
                    data = pivot_df[model].values
                    f.write(f"\n{model.upper()}:\n")
                    f.write(f"  Mean: {np.mean(data):.4f}\n")
                    f.write(f"  Median: {np.median(data):.4f}\n")
                    f.write(f"  Std Dev: {np.std(data, ddof=1):.4f}\n")
                    f.write(f"  Min: {np.min(data):.4f}\n")
                    f.write(f"  Max: {np.max(data):.4f}\n")

                # Friedman test
                f.write("\n\n2. FRIEDMAN TEST\n")
                f.write("-" * 40 + "\n")
                friedman_result = self.friedman_test(metric=metric)

                if friedman_result:
                    f.write(f"Test Statistic: {friedman_result['statistic']:.4f}\n")
                    f.write(f"P-value: {friedman_result['p_value']:.4e}\n")
                    f.write(f"Kendall's W (effect size): {friedman_result['kendalls_w']:.4f}\n")
                    f.write(
                        f"Conclusion: {'Significant differences exist (p < 0.05)' if friedman_result['significant'] else 'No significant differences (p ≥ 0.05)'}\n")

                    if friedman_result['significant']:
                        f.write("\nInterpretation:\n")
                        f.write("There are statistically significant differences in performance between\n")
                        f.write("the models across the datasets. Proceed with post-hoc tests to identify\n")
                        f.write("which specific models differ from each other.\n")

                # Post-hoc tests (only if Friedman is significant)
                if friedman_result.get('significant', False):
                    f.write("\n\n3. POST-HOC TESTS\n")
                    f.write("-" * 40 + "\n")

                    # Nemenyi
                    nemenyi_df = self.nemenyi_post_hoc(metric=metric)
                    sig_pairs = nemenyi_df[nemenyi_df['significant']]
                    f.write(f"\nNemenyi Test Results:\n")
                    f.write(f"  Significant pairs: {len(sig_pairs)} out of {len(nemenyi_df)}\n")

                    if len(sig_pairs) > 0:
                        f.write("\nSignificant differences found between:\n")
                        for _, row in sig_pairs.iterrows():
                            f.write(f"  {row['model_a'].upper()} vs {row['model_b'].upper()}: ")
                            f.write(f"{row['better_model'].upper()} is better ")
                            f.write(f"(rank diff = {row['rank_difference']:.3f})\n")

                    # Wilcoxon with Bonferroni
                    wilcoxon_df = self.wilcoxon_post_hoc_bonferroni(metric=metric)
                    sig_pairs_w = wilcoxon_df[wilcoxon_df['significant']]
                    f.write(f"\n\nWilcoxon Test (Bonferroni-corrected) Results:\n")
                    f.write(f"  Significant pairs: {len(sig_pairs_w)} out of {len(wilcoxon_df)}\n")

                    if len(sig_pairs_w) > 0:
                        f.write("\nSignificant differences found between:\n")
                        for _, row in sig_pairs_w.iterrows():
                            f.write(f"  {row['model_a'].upper()} vs {row['model_b'].upper()}: ")
                            f.write(f"{row['better_model'].upper()} is better ")
                            f.write(f"(p = {row['p_value_bonferroni']:.4e}, ")
                            f.write(f"d = {row['cohens_d']:.3f})\n")

                # Effect sizes
                f.write("\n\n4. EFFECT SIZES\n")
                f.write("-" * 40 + "\n")
                effect_df = self.compute_effect_sizes(metric=metric)

                # Show largest effects
                effect_sorted = effect_df.sort_values('cohens_d', key=abs, ascending=False).head(5)
                f.write("\nTop 5 largest effect sizes:\n")
                for _, row in effect_sorted.iterrows():
                    f.write(f"  {row['model_a'].upper()} vs {row['model_b'].upper()}: ")
                    f.write(f"Cohen's d = {row['cohens_d']:.3f} ({row['effect_size_interpretation']})\n")

                f.write("\n")

        print(f"\n{'=' * 80}")
        print(f"Comprehensive report saved to: {report_path}")
        print(f"{'=' * 80}\n")

    def run_complete_analysis(self, metrics: List[str] = None):
        """
        Run the complete statistical analysis pipeline.
        """
        if metrics is None:
            metrics = ['box_mAP@50', 'box_mAP@50-95', 'box_mean_f1']

        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
        print("=" * 80)

        for metric in metrics:
            print(f"\n{'=' * 80}")
            print(f"ANALYZING METRIC: {metric}")
            print(f"{'=' * 80}")

            # 1. Confidence intervals
            print("\n[1/6] Computing confidence intervals...")
            ci_df = self.compute_aggregated_confidence_intervals(metric=metric)
            self.plot_aggregated_confidence_intervals(ci_df, metric=metric)

            # 2. Friedman test
            print("\n[2/6] Performing Friedman test...")
            friedman_result = self.friedman_test(metric=metric)

            if friedman_result.get('significant', False):
                # 3. Nemenyi post-hoc
                print("\n[3/6] Performing Nemenyi post-hoc test...")
                self.nemenyi_post_hoc(metric=metric)

                # 4. Wilcoxon post-hoc
                print("\n[4/6] Performing Wilcoxon post-hoc test with Bonferroni correction...")
                self.wilcoxon_post_hoc_bonferroni(metric=metric)

                # 5. Critical difference diagram
                print("\n[5/6] Creating critical difference diagram...")
                self.create_critical_difference_diagram(metric=metric)

                # 6. Comprehensive heatmap
                print("\n[6/6] Creating comprehensive significance heatmap...")
                self.create_aggregated_significance_heatmap(metric=metric)
            else:
                print("\n  ⚠ Friedman test not significant - skipping post-hoc tests")
                print("  No significant differences found between models for this metric.")

        # Generate final report
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)
        self.generate_comprehensive_report(metrics=metrics)

        print("\n" + "=" * 80)
        print("✓ COMPLETE STATISTICAL ANALYSIS FINISHED!")
        print(f"  All results saved to: {self.output_dir}")
        print("=" * 80 + "\n")


class AggregatedFailureModeAnalyzer:
    """
    Aggregated Failure Mode Analysis across multiple datasets
    """

    def __init__(self, datasets_info: List[Dict], models: List[str],
                 output_dir: str, iou_threshold: float = 0.5):
        self.datasets_info = datasets_info
        self.models = models
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold

        os.makedirs(output_dir, exist_ok=True)

        self.failure_modes = {
            'missed_detection': 'Ground truth object not detected',
            'background_fp': 'False positive on background',
            'boundary_error': 'Incorrect bounding box size/location',
            'class_confusion': 'Wrong class predicted',
            'duplicate_detection': 'Multiple predictions for same object',
            'split_detection': 'Object split into multiple detections'
        }

    def analyze_image(self, gt_path: str, inf_path: str, img_width: int, img_height: int) -> Dict:
        """Analyze failure modes for a single image"""
        gt_boxes = load_yolo_boxes(gt_path)
        pred_boxes = load_inference_boxes(inf_path, img_width, img_height)

        failures = defaultdict(int)

        # Match predictions to ground truth
        matched_pred = set()

        for gt_box in gt_boxes:
            # Find all predictions that match this GT
            matches = []
            for i, pred_box in enumerate(pred_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > 0:
                    matches.append((i, iou, pred_box.cls == gt_box.cls))

            if len(matches) == 0:
                # Missed detection
                failures['missed_detection'] += 1
            else:
                # Sort by IoU
                matches.sort(key=lambda x: x[1], reverse=True)
                best_match = matches[0]

                if best_match[1] < self.iou_threshold:
                    # Poor localization
                    failures['boundary_error'] += 1

                if not best_match[2]:
                    # Wrong class
                    failures['class_confusion'] += 1

                # Check for duplicates
                high_iou_matches = [m for m in matches if m[1] >= self.iou_threshold]
                if len(high_iou_matches) > 1:
                    failures['duplicate_detection'] += len(high_iou_matches) - 1

                # Mark predictions as matched
                for pred_idx, _, _ in high_iou_matches:
                    matched_pred.add(pred_idx)

        # Analyze unmatched predictions (false positives)
        for i, pred_box in enumerate(pred_boxes):
            if i not in matched_pred:
                # Check if it overlaps with any ground truth
                has_overlap = False
                for gt_box in gt_boxes:
                    if calculate_iou(pred_box, gt_box) > 0.1:
                        has_overlap = True
                        break

                if has_overlap:
                    failures['split_detection'] += 1
                else:
                    failures['background_fp'] += 1

        return dict(failures)

    def analyze_single_dataset(self, dataset_info: Dict) -> pd.DataFrame:
        """Analyze failure modes for a single dataset"""
        dataset_name = dataset_info['name']
        gt_folder = dataset_info['gt_folder']
        inference_root = dataset_info['inference_root']
        image_folder = dataset_info['image_folder']

        print(f"  Analyzing dataset: {dataset_name}")

        image_files = [f.replace('.txt', '') for f in os.listdir(gt_folder)
                       if f.endswith('.txt')]

        all_results = []

        for model in self.models:
            model_failures = defaultdict(int)

            for img_name in image_files:
                # Get image dimensions
                image_path = os.path.join(image_folder, f"{img_name}.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{img_name}.png")

                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        img_width, img_height = 640, 640
                else:
                    img_width, img_height = 640, 640

                gt_path = os.path.join(gt_folder, f"{img_name}.txt")
                inf_path = os.path.join(inference_root, model, f"{img_name}.txt")

                img_failures = self.analyze_image(gt_path, inf_path, img_width, img_height)

                for mode, count in img_failures.items():
                    model_failures[mode] += count

            # Add results - ensure ALL failure modes are present, even if zero
            result = {
                'dataset': dataset_name,
                'model': model
            }
            # Add all failure modes with explicit 0 if not present
            for mode in self.failure_modes.keys():
                result[mode] = model_failures.get(mode, 0)

            # Calculate percentages
            total_failures = sum(result[mode] for mode in self.failure_modes.keys())
            for mode in self.failure_modes.keys():
                count = result[mode]
                result[f'{mode}_pct'] = (count / total_failures * 100) if total_failures > 0 else 0

            all_results.append(result)

        return pd.DataFrame(all_results)

    def analyze_all_datasets(self) -> pd.DataFrame:
        """Analyze all datasets and aggregate results"""
        all_dfs = []

        for dataset_info in self.datasets_info:
            df = self.analyze_single_dataset(dataset_info)
            all_dfs.append(df)

        # Combine all results
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Save combined results
        output_path = os.path.join(self.output_dir, 'failure_modes_all_datasets.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\nAggregated failure mode results saved to: {output_path}")

        # Also save aggregated summary (summed across datasets)
        aggregated_df = self._aggregate_across_datasets(combined_df)
        agg_output_path = os.path.join(self.output_dir, 'failure_modes_aggregated_summary.csv')
        aggregated_df.to_csv(agg_output_path, index=False)
        print(f"Aggregated summary saved to: {agg_output_path}")

        return combined_df

    def _aggregate_across_datasets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate failure counts across all datasets"""
        # Only include columns that exist in the dataframe
        available_failure_modes = [mode for mode in self.failure_modes.keys() if mode in df.columns]

        if not available_failure_modes:
            print("  ⚠️  WARNING: No failure mode columns found in dataframe")
            return pd.DataFrame()

        failure_cols = [col for col in df.columns
                        if col not in ['dataset', 'model'] and not col.endswith('_pct')]

        # Group by model and sum failure counts
        grouped = df.groupby('model')[failure_cols].sum().reset_index()

        # Recalculate percentages for available failure modes only
        if available_failure_modes:
            # Calculate total using only available failure modes
            grouped['total_failures'] = grouped[
                [mode for mode in available_failure_modes if mode in grouped.columns]].sum(axis=1)

            for mode in available_failure_modes:
                if mode in grouped.columns:
                    grouped[f'{mode}_pct'] = (grouped[mode] / grouped['total_failures'] * 100).fillna(0)

        return grouped

    def plot_aggregated_failure_modes(self, df: pd.DataFrame):
        """Visualize aggregated failure mode distributions"""
        # Aggregate across datasets
        agg_df = self._aggregate_across_datasets(df)

        # Order models and add display names
        agg_df = agg_df.copy()
        ordered_models = get_ordered_models(agg_df['model'].unique())
        agg_df['model'] = pd.Categorical(agg_df['model'], categories=ordered_models, ordered=True)
        agg_df = agg_df.sort_values('model')
        agg_df['display_name'] = agg_df['model'].apply(get_display_name)

        failure_cols = [col for col in agg_df.columns if col in self.failure_modes.keys()]

        if len(failure_cols) == 0:
            return

        # Plot 1: Stacked bar chart (aggregated)
        plt.figure(figsize=(12, 6))
        df_plot = agg_df.set_index('display_name')[failure_cols]
        df_plot.plot(kind='bar', stacked=True, figsize=(12, 6),
                     colormap='Set3', edgecolor='black', linewidth=0.5)

        plt.title('Failure Mode Distribution by Model (All Datasets)', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Number of Failures', fontsize=12)
        plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'failure_modes_stacked_aggregated.png'), dpi=300)
        plt.close()

        # Plot 2: Per-dataset comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Order df for per-dataset plots
        df = df.copy()
        df['model'] = pd.Categorical(df['model'], categories=ordered_models, ordered=True)
        df = df.sort_values('model')
        df['display_name'] = df['model'].apply(get_display_name)

        datasets = df['dataset'].unique()
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break

            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset].set_index('display_name')[failure_cols]

            dataset_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', edgecolor='black', linewidth=0.5)

            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Failures', fontsize=10)
            ax.legend().set_visible(False)
            ax.tick_params(axis='x', rotation=45, labelsize=8)

        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)

        # Add legend to the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Failure Mode', loc='center left', bbox_to_anchor=(0.95, 0.5))

        plt.suptitle('Failure Modes - Per Dataset Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(self.output_dir, 'failure_modes_per_dataset.png'), dpi=300)
        plt.close()

        print(f"Aggregated failure mode plots saved to: {self.output_dir}")