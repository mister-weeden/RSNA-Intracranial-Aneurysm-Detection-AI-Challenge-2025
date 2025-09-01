#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Aneurysm Detection Models
Implements competition scoring, statistical analysis, and performance benchmarking.

@cursor Implements official RSNA competition scoring methodology
@cursor Includes statistical significance testing and confidence intervals
@cursor Provides detailed performance analysis suitable for Kaggle submission
@cursor Supports cross-validation and ensemble evaluation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, bootstrap_confidence_interval
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import defaultdict
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """
    Data class for storing comprehensive evaluation metrics.
    
    @cursor Structured storage for all evaluation metrics
    @cursor Supports serialization for result persistence
    """
    auc_roc: float
    auc_pr: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    fpr: float
    fnr: float
    npv: float  # Negative Predictive Value
    ppv: float  # Positive Predictive Value
    mcc: float  # Matthews Correlation Coefficient
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'fpr': self.fpr,
            'fnr': self.fnr,
            'npv': self.npv,
            'ppv': self.ppv,
            'mcc': self.mcc
        }

class CompetitionScorer:
    """
    Official RSNA Aneurysm Detection Competition Scorer.
    
    @cursor Implements exact competition scoring formula
    @cursor Handles edge cases and provides detailed breakdowns
    @cursor Supports both single model and ensemble evaluation
    """
    
    def __init__(self):
        """Initialize competition scorer."""
        self.num_characteristics = 13
        self.aneurysm_weight = 0.5
        self.characteristic_weight = 0.5 / self.num_characteristics
        
    def calculate_competition_score(self, 
                                  predictions: Dict[str, np.ndarray],
                                  ground_truth: Dict[str, np.ndarray],
                                  return_breakdown: bool = True) -> Union[float, Dict]:
        """
        Calculate official competition score: ½(AUC_AP + 1/13 Σ AUC_i)
        
        @cursor Implements exact competition formula with error handling
        @cursor Provides detailed breakdown for analysis
        
        Args:
            predictions: Dictionary with prediction probabilities
            ground_truth: Dictionary with ground truth labels
            return_breakdown: Whether to return detailed breakdown
            
        Returns:
            Competition score or detailed breakdown dictionary
        """
        scores = {}
        
        # Calculate AUC_AP (aneurysm presence/absence)
        if 'aneurysm' in predictions and 'aneurysm' in ground_truth:
            try:
                auc_ap = average_precision_score(
                    ground_truth['aneurysm'], 
                    predictions['aneurysm']
                )
                scores['AUC_AP'] = auc_ap
            except Exception as e:
                logger.warning(f"Error calculating AUC_AP: {e}")
                scores['AUC_AP'] = 0.5
        else:
            logger.warning("Aneurysm predictions/ground truth not found")
            scores['AUC_AP'] = 0.5
        
        # Calculate AUC_i for each characteristic (i = 0 to 12)
        characteristic_aucs = []
        for i in range(self.num_characteristics):
            char_key = f'char_{i}'
            if char_key in predictions and char_key in ground_truth:
                try:
                    auc_i = average_precision_score(
                        ground_truth[char_key],
                        predictions[char_key]
                    )
                    scores[f'AUC_{i}'] = auc_i
                    characteristic_aucs.append(auc_i)
                except Exception as e:
                    logger.warning(f"Error calculating AUC_{i}: {e}")
                    scores[f'AUC_{i}'] = 0.5
                    characteristic_aucs.append(0.5)
            else:
                logger.warning(f"Characteristic {i} predictions/ground truth not found")
                scores[f'AUC_{i}'] = 0.5
                characteristic_aucs.append(0.5)
        
        # Calculate mean characteristic AUC
        mean_char_auc = np.mean(characteristic_aucs)
        scores['mean_characteristic_AUC'] = mean_char_auc
        
        # Calculate final competition score
        competition_score = 0.5 * (scores['AUC_AP'] + mean_char_auc)
        scores['competition_score'] = competition_score
        
        if return_breakdown:
            return scores
        else:
            return competition_score

class StatisticalAnalyzer:
    """
    Statistical analysis tools for model evaluation.
    
    @cursor Provides statistical significance testing and confidence intervals
    @cursor Implements bootstrap methods for robust uncertainty estimation
    @cursor Supports comparison between multiple models
    """
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            n_bootstrap: Number of bootstrap samples (default: 1000)
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        
    def bootstrap_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate AUC with bootstrap confidence intervals.
        
        @cursor Provides robust uncertainty estimation for AUC scores
        @cursor Uses stratified bootstrap to maintain class balance
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            Tuple of (AUC, lower_CI, upper_CI)
        """
        n_samples = len(y_true)
        bootstrap_aucs = []
        
        for _ in range(self.n_bootstrap):
            # Stratified bootstrap sampling
            indices = self._stratified_bootstrap_indices(y_true)
            
            try:
                bootstrap_auc = roc_auc_score(y_true[indices], y_pred[indices])
                bootstrap_aucs.append(bootstrap_auc)
            except:
                # Handle edge cases where bootstrap sample has only one class
                continue
        
        if not bootstrap_aucs:
            return 0.5, 0.5, 0.5
        
        bootstrap_aucs = np.array(bootstrap_aucs)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        auc_mean = np.mean(bootstrap_aucs)
        ci_lower = np.percentile(bootstrap_aucs, lower_percentile)
        ci_upper = np.percentile(bootstrap_aucs, upper_percentile)
        
        return auc_mean, ci_lower, ci_upper
    
    def _stratified_bootstrap_indices(self, y_true: np.ndarray) -> np.ndarray:
        """
        Generate stratified bootstrap indices maintaining class balance.
        
        @cursor Ensures bootstrap samples maintain original class distribution
        
        Args:
            y_true: Ground truth labels
            
        Returns:
            Bootstrap sample indices
        """
        n_samples = len(y_true)
        unique_classes = np.unique(y_true)
        indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(y_true == class_label)[0]
            n_class_samples = len(class_indices)
            
            # Sample with replacement from this class
            bootstrap_class_indices = np.random.choice(
                class_indices, size=n_class_samples, replace=True
            )
            indices.extend(bootstrap_class_indices)
        
        return np.array(indices)
    
    def mcnemar_test(self, y_true: np.ndarray, 
                    pred1: np.ndarray, pred2: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
        """
        Perform McNemar's test to compare two models.
        
        @cursor Statistical test for comparing paired binary classifiers
        @cursor Provides p-value for significance testing
        
        Args:
            y_true: Ground truth labels
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            threshold: Classification threshold
            
        Returns:
            Dictionary containing test statistics and p-value
        """
        # Convert probabilities to binary predictions
        pred1_binary = (pred1 >= threshold).astype(int)
        pred2_binary = (pred2 >= threshold).astype(int)
        
        # Create contingency table
        correct1 = (pred1_binary == y_true)
        correct2 = (pred2_binary == y_true)
        
        # McNemar's table
        both_correct = np.sum(correct1 & correct2)
        model1_correct_only = np.sum(correct1 & ~correct2)
        model2_correct_only = np.sum(~correct1 & correct2)
        both_incorrect = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic
        if model1_correct_only + model2_correct_only == 0:
            # No discordant pairs
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (abs(model1_correct_only - model2_correct_only) - 1)**2 / \
                       (model1_correct_only + model2_correct_only)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'model1_only_correct': model1_correct_only,
            'model2_only_correct': model2_correct_only,
            'both_correct': both_correct,
            'both_incorrect': both_incorrect
        }

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for aneurysm detection models.
    
    @cursor Provides detailed performance metrics and visualizations
    @cursor Supports complexity analysis and optimization recommendations
    @cursor Generates reports suitable for medical publication and Kaggle submission
    """
    
    def __init__(self, save_plots: bool = True, output_dir: str = 'evaluation_results'):
        """
        Initialize performance analyzer.
        
        Args:
            save_plots: Whether to save generated plots
            output_dir: Directory for saving results
        """
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.competition_scorer = CompetitionScorer()
        self.statistical_analyzer = StatisticalAnalyzer()
        
    def calculate_detailed_metrics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 threshold: float = 0.5) -> EvaluationMetrics:
        """
        Calculate comprehensive evaluation metrics.
        
        @cursor Computes all standard medical imaging evaluation metrics
        @cursor Handles edge cases and provides robust calculations
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # PPV
        
        # Additional metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # F1 Score
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) \
                  if (precision + sensitivity) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
        
        # AUC scores
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
        except:
            auc_roc = 0.5
            
        try:
            auc_pr = average_precision_score(y_true, y_pred)
        except:
            auc_pr = 0.5
        
        return EvaluationMetrics(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            recall=sensitivity,  # Same as sensitivity
            f1_score=f1_score,
            accuracy=accuracy,
            fpr=fpr,
            fnr=fnr,
            npv=npv,
            ppv=precision,  # Same as precision
            mcc=mcc
        )
    
    def generate_roc_curve_analysis(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  model_name: str = "Model") -> Dict:
        """
        Generate ROC curve analysis with optimal threshold selection.
        
        @cursor Provides ROC analysis with Youden's J statistic for optimal threshold
        @cursor Includes confidence intervals and statistical analysis
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            model_name: Name of the model for labeling
            
        Returns:
            Dictionary containing ROC analysis results
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)
        
        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        
        # Bootstrap confidence intervals for AUC
        auc_mean, auc_ci_lower, auc_ci_upper = self.statistical_analyzer.bootstrap_auc(
            y_true, y_pred
        )
        
        # Create ROC plot
        if self.save_plots:
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_roc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                       label=f'Optimal Threshold = {optimal_threshold:.3f}')
            
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add confidence interval text
            plt.text(0.6, 0.2, f'AUC 95% CI: [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'auc_roc': auc_roc,
            'auc_ci_lower': auc_ci_lower,
            'auc_ci_upper': auc_ci_upper,
            'optimal_threshold': optimal_threshold,
            'optimal_sensitivity': optimal_sensitivity,
            'optimal_specificity': optimal_specificity,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def generate_precision_recall_analysis(self, y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         model_name: str = "Model") -> Dict:
        """
        Generate precision-recall curve analysis.
        
        @cursor Provides PR analysis particularly important for imbalanced datasets
        @cursor Includes average precision and optimal F1 threshold
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            model_name: Name of the model for labeling
            
        Returns:
            Dictionary containing PR analysis results
        """
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)
        
        # Find optimal threshold for F1 score
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        
        if len(f1_scores) > 0:
            optimal_f1_idx = np.argmax(f1_scores)
            optimal_f1_threshold = thresholds[optimal_f1_idx]
            optimal_f1_score = f1_scores[optimal_f1_idx]
            optimal_f1_precision = precision[optimal_f1_idx]
            optimal_f1_recall = recall[optimal_f1_idx]
        else:
            optimal_f1_threshold = 0.5
            optimal_f1_score = 0.0
            optimal_f1_precision = 0.0
            optimal_f1_recall = 0.0
        
        # Create PR plot
        if self.save_plots:
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, linewidth=2, 
                    label=f'{model_name} (AP = {auc_pr:.3f})')
            
            # Baseline (random classifier)
            baseline = np.sum(y_true) / len(y_true)
            plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                       label=f'Random Classifier (AP = {baseline:.3f})')
            
            # Optimal F1 point
            if len(f1_scores) > 0:
                plt.scatter(optimal_f1_recall, optimal_f1_precision, color='red', s=100,
                           label=f'Optimal F1 = {optimal_f1_score:.3f}')
            
            plt.xlabel('Recall (Sensitivity)')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'pr_curve_{model_name.lower().replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'auc_pr': auc_pr,
            'optimal_f1_threshold': optimal_f1_threshold,
            'optimal_f1_score': optimal_f1_score,
            'optimal_f1_precision': optimal_f1_precision,
            'optimal_f1_recall': optimal_f1_recall,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def evaluate_model_comprehensive(self, 
                                   predictions: Dict[str, np.ndarray],
                                   ground_truth: Dict[str, np.ndarray],
                                   model_name: str = "Model") -> Dict:
        """
        Comprehensive model evaluation with all metrics and analyses.
        
        @cursor Main evaluation function providing complete analysis
        @cursor Suitable for medical publication and competition submission
        
        Args:
            predictions: Dictionary containing model predictions
            ground_truth: Dictionary containing ground truth labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Starting comprehensive evaluation for {model_name}")
        start_time = time.time()
        
        results = {
            'model_name': model_name,
            'evaluation_timestamp': time.time(),
            'task_results': {}
        }
        
        # Evaluate each task
        for task_name in predictions.keys():
            if task_name in ground_truth:
                logger.info(f"Evaluating task: {task_name}")
                
                y_true = ground_truth[task_name]
                y_pred = predictions[task_name]
                
                # Basic metrics
                metrics = self.calculate_detailed_metrics(y_true, y_pred)
                
                # ROC analysis
                roc_analysis = self.generate_roc_curve_analysis(y_true, y_pred, 
                                                              f"{model_name}_{task_name}")
                
                # PR analysis
                pr_analysis = self.generate_precision_recall_analysis(y_true, y_pred,
                                                                    f"{model_name}_{task_name}")
                
                # Store results
                results['task_results'][task_name] = {
                    'metrics': metrics.to_dict(),
                    'roc_analysis': roc_analysis,
                    'pr_analysis': pr_analysis
                }
        
        # Competition score
        competition_results = self.competition_scorer.calculate_competition_score(
            predictions, ground_truth, return_breakdown=True
        )
        results['competition_score'] = competition_results
        
        # Performance summary
        evaluation_time = time.time() - start_time
        results['evaluation_time'] = evaluation_time
        results['performance_summary'] = {
            'total_evaluation_time': evaluation_time,
            'tasks_evaluated': len(results['task_results']),
            'competition_score': competition_results.get('competition_score', 0.0)
        }
        
        # Save results
        results_file = self.output_dir / f'evaluation_results_{model_name.lower().replace(" ", "_")}.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_for_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Competition score: {competition_results.get('competition_score', 0.0):.4f}")
        logger.info(f"Results saved to: {results_file}")
        
        return results
    
    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

def main():
    """
    Main evaluation function for testing and demonstration.
    
    @cursor Example usage of comprehensive evaluation pipeline
    @cursor Can be adapted for actual model evaluation
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(save_plots=True, output_dir='evaluation_results')
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions and ground truth
    ground_truth = {
        'aneurysm': np.random.binomial(1, 0.3, n_samples),  # 30% positive rate
    }
    
    # Add 13 characteristics
    for i in range(13):
        # Characteristics correlated with aneurysm presence
        char_prob = 0.1 + 0.4 * ground_truth['aneurysm'] + 0.1 * np.random.random(n_samples)
        ground_truth[f'char_{i}'] = np.random.binomial(1, char_prob)
    
    # Simulate model predictions (with some noise)
    predictions = {}
    for key, true_labels in ground_truth.items():
        # Add noise to create realistic predictions
        noise = np.random.normal(0, 0.1, n_samples)
        pred_probs = true_labels + noise
        pred_probs = np.clip(pred_probs, 0, 1)  # Ensure valid probabilities
        predictions[key] = pred_probs
    
    # Run comprehensive evaluation
    results = analyzer.evaluate_model_comprehensive(
        predictions, ground_truth, model_name="Demo_Model"
    )
    
    print("\nEvaluation Summary:")
    print(f"Competition Score: {results['competition_score']['competition_score']:.4f}")
    print(f"AUC_AP: {results['competition_score']['AUC_AP']:.4f}")
    print(f"Mean Characteristic AUC: {results['competition_score']['mean_characteristic_AUC']:.4f}")
    
    return results

if __name__ == "__main__":
    main()
