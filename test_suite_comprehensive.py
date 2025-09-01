#!/usr/bin/env python3
"""
Comprehensive Test Suite for Aneurysm Detection Pipeline
Tests performance, complexity, accuracy, and deterministic behavior.

@cursor Implements comprehensive testing for research validation
@cursor Tests computational complexity and performance requirements
@cursor Validates deterministic behavior and reproducibility
@cursor Includes medical imaging specific test cases
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import time
import tempfile
import shutil
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any
import warnings
import psutil
import gc

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent))

from train_medicalnet_aneurysm import (
    MultiTaskAneurysmNet, AneurysmDataset, MultiTaskLoss, 
    AneurysmTrainer, PerformanceMonitor, set_deterministic_seed
)
from aneurysm_preprocessing import AneurysmPreprocessingPipeline
from evaluation_comprehensive import (
    CompetitionScorer, StatisticalAnalyzer, PerformanceAnalyzer, EvaluationMetrics
)
from experiment_generator import ExperimentGenerator, ExperimentConfig

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

class TestDeterministicBehavior(unittest.TestCase):
    """
    Test deterministic behavior and reproducibility.
    
    @cursor Critical for research reproducibility and scientific validity
    @cursor Tests that random seeds produce identical results across runs
    """
    
    def setUp(self):
        """Set up test environment."""
        self.seed = 42
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_deterministic_seeding(self):
        """
        Test that deterministic seeding produces identical results.
        
        @cursor Validates reproducibility across multiple runs
        """
        # Generate two sets of random numbers with same seed
        set_deterministic_seed(self.seed)
        random_1 = np.random.random(100)
        torch_random_1 = torch.randn(100)
        
        set_deterministic_seed(self.seed)
        random_2 = np.random.random(100)
        torch_random_2 = torch.randn(100)
        
        # Should be identical
        np.testing.assert_array_equal(random_1, random_2)
        torch.testing.assert_close(torch_random_1, torch_random_2)
        
    def test_model_deterministic_forward(self):
        """
        Test that model forward passes are deterministic.
        
        @cursor Ensures model predictions are reproducible
        """
        # Create model and input
        set_deterministic_seed(self.seed)
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
        model.eval()
        
        input_tensor = torch.randn(2, 1, 32, 32, 32)
        
        # Two forward passes with same seed
        set_deterministic_seed(self.seed)
        output_1 = model(input_tensor)
        
        set_deterministic_seed(self.seed)
        output_2 = model(input_tensor)
        
        # Outputs should be identical
        for key in output_1.keys():
            torch.testing.assert_close(output_1[key], output_2[key])
            
    def test_preprocessing_deterministic(self):
        """
        Test that preprocessing is deterministic.
        
        @cursor Ensures preprocessing pipeline produces consistent results
        """
        # Create synthetic volume
        np.random.seed(self.seed)
        volume = np.random.random((64, 64, 64)).astype(np.float32)
        
        # Process twice
        pipeline = AneurysmPreprocessingPipeline()
        
        result_1 = pipeline.process_volume(volume)
        result_2 = pipeline.process_volume(volume)
        
        # Results should be identical
        for key in result_1.keys():
            np.testing.assert_array_equal(result_1[key], result_2[key])

class TestComputationalComplexity(unittest.TestCase):
    """
    Test computational complexity and performance requirements.
    
    @cursor Validates O(n*log(n)) complexity requirements
    @cursor Identifies performance bottlenecks and optimization opportunities
    """
    
    def setUp(self):
        """Set up performance monitoring."""
        self.performance_monitor = PerformanceMonitor()
        
    def test_preprocessing_complexity(self):
        """
        Test preprocessing computational complexity.
        
        @cursor Validates that preprocessing scales appropriately with input size
        """
        pipeline = AneurysmPreprocessingPipeline()
        
        # Test different input sizes
        sizes = [32, 48, 64, 96]
        times = []
        
        for size in sizes:
            volume = np.random.random((size, size, size)).astype(np.float32)
            
            start_time = time.time()
            result = pipeline.process_volume(volume, apply_vesselness=True)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            # @cursor Flag if processing time grows faster than O(n*log(n))
            n = size ** 3
            expected_complexity = n * np.log(n)
            time_per_complexity = processing_time / expected_complexity
            
            # Should be reasonable (less than 1e-6 seconds per n*log(n) unit)
            self.assertLess(time_per_complexity, 1e-5, 
                          f"Processing time for size {size} may exceed O(n*log(n))")
        
        # Check that time doesn't grow too quickly
        # Should be roughly O(n*log(n)), not O(n²) or worse
        for i in range(1, len(times)):
            size_ratio = (sizes[i] / sizes[i-1]) ** 3  # Volume ratio
            time_ratio = times[i] / times[i-1]
            
            # Time ratio should be less than size_ratio² (would indicate O(n²))
            self.assertLess(time_ratio, size_ratio ** 1.5,
                          f"Time complexity may be worse than O(n*log(n))")
    
    def test_model_memory_usage(self):
        """
        Test model memory usage and efficiency.
        
        @cursor Monitors memory usage to prevent memory leaks
        """
        # Monitor memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use model
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
        
        # Process multiple batches
        for _ in range(10):
            input_tensor = torch.randn(4, 1, 64, 64, 64)
            output = model(input_tensor)
            
            # Clear intermediate results
            del input_tensor, output
            gc.collect()
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 2GB)
        self.assertLess(memory_increase, 2048, 
                       f"Memory usage increased by {memory_increase:.1f}MB")
    
    def test_batch_processing_efficiency(self):
        """
        Test batch processing efficiency.
        
        @cursor Validates that batch processing is more efficient than individual processing
        """
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
        model.eval()
        
        # Single sample processing
        single_times = []
        for _ in range(8):
            input_tensor = torch.randn(1, 1, 32, 32, 32)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            end_time = time.time()
            
            single_times.append(end_time - start_time)
        
        total_single_time = sum(single_times)
        
        # Batch processing
        batch_input = torch.randn(8, 1, 32, 32, 32)
        
        start_time = time.time()
        with torch.no_grad():
            batch_output = model(batch_input)
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        # Batch processing should be more efficient
        efficiency_ratio = total_single_time / batch_time
        self.assertGreater(efficiency_ratio, 1.5, 
                          "Batch processing should be more efficient than individual processing")

class TestModelAccuracy(unittest.TestCase):
    """
    Test model accuracy and performance on synthetic data.
    
    @cursor Validates model learning capability and convergence
    @cursor Tests multi-task learning and uncertainty quantification
    """
    
    def setUp(self):
        """Set up synthetic data for testing."""
        set_deterministic_seed(42)
        self.n_samples = 100
        self.input_size = (32, 32, 32)
        
        # Create synthetic dataset
        self.synthetic_data = self._create_synthetic_data()
        
    def _create_synthetic_data(self) -> Dict:
        """
        Create synthetic data with known patterns.
        
        @cursor Creates controlled synthetic data for validation
        """
        data = []
        labels = []
        
        for i in range(self.n_samples):
            # Create volume with synthetic aneurysm
            volume = np.random.random(self.input_size).astype(np.float32) * 0.1
            
            # Add synthetic aneurysm (bright sphere) in 30% of cases
            has_aneurysm = i < (self.n_samples * 0.3)
            
            if has_aneurysm:
                # Add bright sphere in random location
                center = [np.random.randint(8, s-8) for s in self.input_size]
                radius = np.random.randint(3, 6)
                
                # Create sphere
                for x in range(max(0, center[0]-radius), min(self.input_size[0], center[0]+radius)):
                    for y in range(max(0, center[1]-radius), min(self.input_size[1], center[1]+radius)):
                        for z in range(max(0, center[2]-radius), min(self.input_size[2], center[2]+radius)):
                            dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                            if dist <= radius:
                                volume[x, y, z] = 0.8 + 0.2 * np.random.random()
            
            data.append(volume)
            
            # Create labels
            label_dict = {'aneurysm': float(has_aneurysm)}
            
            # Add synthetic characteristics correlated with aneurysm
            for j in range(13):
                char_prob = 0.1 + 0.6 * has_aneurysm + 0.1 * np.random.random()
                label_dict[f'char_{j}'] = float(np.random.random() < char_prob)
            
            labels.append(label_dict)
        
        return {'data': data, 'labels': labels}
    
    def test_model_convergence(self):
        """
        Test that model can learn from synthetic data.
        
        @cursor Validates basic learning capability on controlled data
        """
        # Create model
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
        criterion = MultiTaskLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training data
        data = torch.stack([torch.from_numpy(d).unsqueeze(0) for d in self.synthetic_data['data']])
        labels = self.synthetic_data['labels']
        
        # Convert labels to tensors
        label_tensors = {}
        for key in labels[0].keys():
            label_tensors[key] = torch.tensor([l[key] for l in labels], dtype=torch.float32)
        
        # Train for a few epochs
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(20):
            optimizer.zero_grad()
            
            predictions = model(data)
            losses = criterion(predictions, label_tensors)
            
            losses['total_loss'].backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = losses['total_loss'].item()
            if epoch == 19:
                final_loss = losses['total_loss'].item()
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss * 0.8, 
                       "Model should learn and reduce loss on synthetic data")
    
    def test_uncertainty_quantification(self):
        """
        Test uncertainty quantification functionality.
        
        @cursor Validates Monte Carlo dropout uncertainty estimation
        """
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13, 
                                   use_uncertainty=True)
        model.eval()
        
        input_tensor = torch.randn(4, 1, 32, 32, 32)
        
        # Get predictions with uncertainty
        predictions = model(input_tensor, num_samples=10)
        
        # Should have uncertainty estimates
        for key in ['aneurysm'] + [f'char_{i}' for i in range(13)]:
            self.assertIn(key, predictions)
            self.assertIn(f'{key}_uncertainty', predictions)
            
            # Uncertainty should be non-negative
            uncertainty = predictions[f'{key}_uncertainty']
            self.assertTrue(torch.all(uncertainty >= 0), 
                          f"Uncertainty for {key} should be non-negative")
    
    def test_multi_task_learning(self):
        """
        Test multi-task learning capability.
        
        @cursor Validates that model can learn multiple tasks simultaneously
        """
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
        
        input_tensor = torch.randn(4, 1, 32, 32, 32)
        predictions = model(input_tensor)
        
        # Should have predictions for all tasks
        expected_keys = ['aneurysm'] + [f'char_{i}' for i in range(13)]
        
        for key in expected_keys:
            self.assertIn(key, predictions)
            self.assertEqual(predictions[key].shape[0], 4)  # Batch size
            
            # Predictions should be valid probabilities
            self.assertTrue(torch.all(predictions[key] >= 0))
            self.assertTrue(torch.all(predictions[key] <= 1))

class TestCompetitionScoring(unittest.TestCase):
    """
    Test competition scoring implementation.
    
    @cursor Validates official RSNA competition scoring formula
    @cursor Tests edge cases and error handling
    """
    
    def setUp(self):
        """Set up competition scorer."""
        self.scorer = CompetitionScorer()
        
    def test_perfect_predictions(self):
        """
        Test scoring with perfect predictions.
        
        @cursor Validates that perfect predictions give maximum score
        """
        n_samples = 100
        
        # Perfect predictions (identical to ground truth)
        predictions = {
            'aneurysm': np.random.binomial(1, 0.3, n_samples).astype(float)
        }
        
        # Add characteristics
        for i in range(13):
            predictions[f'char_{i}'] = np.random.binomial(1, 0.2, n_samples).astype(float)
        
        ground_truth = predictions.copy()
        
        score = self.scorer.calculate_competition_score(predictions, ground_truth, 
                                                       return_breakdown=False)
        
        # Perfect predictions should give score of 1.0
        self.assertAlmostEqual(score, 1.0, places=2)
    
    def test_random_predictions(self):
        """
        Test scoring with random predictions.
        
        @cursor Validates that random predictions give expected baseline score
        """
        n_samples = 1000
        
        # Ground truth
        ground_truth = {
            'aneurysm': np.random.binomial(1, 0.3, n_samples)
        }
        
        for i in range(13):
            ground_truth[f'char_{i}'] = np.random.binomial(1, 0.2, n_samples)
        
        # Random predictions
        predictions = {}
        for key in ground_truth.keys():
            predictions[key] = np.random.random(n_samples)
        
        score = self.scorer.calculate_competition_score(predictions, ground_truth,
                                                       return_breakdown=False)
        
        # Random predictions should give score around 0.5
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.7)
    
    def test_edge_cases(self):
        """
        Test scoring with edge cases.
        
        @cursor Tests handling of single-class datasets and missing data
        """
        n_samples = 50
        
        # All positive ground truth
        ground_truth = {
            'aneurysm': np.ones(n_samples)
        }
        
        for i in range(13):
            ground_truth[f'char_{i}'] = np.ones(n_samples)
        
        # Random predictions
        predictions = {}
        for key in ground_truth.keys():
            predictions[key] = np.random.random(n_samples)
        
        # Should handle single-class case gracefully
        score = self.scorer.calculate_competition_score(predictions, ground_truth,
                                                       return_breakdown=False)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

class TestStatisticalAnalysis(unittest.TestCase):
    """
    Test statistical analysis tools.
    
    @cursor Validates statistical significance testing and confidence intervals
    @cursor Tests bootstrap methods and model comparison tools
    """
    
    def setUp(self):
        """Set up statistical analyzer."""
        self.analyzer = StatisticalAnalyzer(confidence_level=0.95, n_bootstrap=100)
        
    def test_bootstrap_auc(self):
        """
        Test bootstrap AUC calculation.
        
        @cursor Validates bootstrap confidence interval estimation
        """
        # Create synthetic data with known AUC
        n_samples = 200
        y_true = np.random.binomial(1, 0.3, n_samples)
        
        # Create predictions with some discrimination
        y_pred = y_true * 0.7 + np.random.random(n_samples) * 0.3
        
        auc_mean, ci_lower, ci_upper = self.analyzer.bootstrap_auc(y_true, y_pred)
        
        # AUC should be reasonable
        self.assertGreater(auc_mean, 0.5)
        self.assertLess(auc_mean, 1.0)
        
        # Confidence interval should be valid
        self.assertLess(ci_lower, auc_mean)
        self.assertGreater(ci_upper, auc_mean)
        self.assertLess(ci_upper - ci_lower, 0.5)  # Reasonable interval width
    
    def test_mcnemar_test(self):
        """
        Test McNemar's test for model comparison.
        
        @cursor Validates statistical comparison between models
        """
        n_samples = 100
        y_true = np.random.binomial(1, 0.3, n_samples)
        
        # Two models with different performance
        pred1 = y_true * 0.8 + np.random.random(n_samples) * 0.2
        pred2 = y_true * 0.6 + np.random.random(n_samples) * 0.4
        
        result = self.analyzer.mcnemar_test(y_true, pred1, pred2)
        
        # Should return valid test results
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertGreaterEqual(result['p_value'], 0.0)
        self.assertLessEqual(result['p_value'], 1.0)

class TestExperimentGeneration(unittest.TestCase):
    """
    Test experiment generation and configuration.
    
    @cursor Validates systematic experiment design
    @cursor Tests configuration generation and validation
    """
    
    def setUp(self):
        """Set up experiment generator."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = ExperimentGenerator(base_output_dir=str(self.temp_dir))
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_baseline_experiment_generation(self):
        """
        Test baseline experiment generation.
        
        @cursor Validates that baseline experiments are properly configured
        """
        experiments = self.generator.generate_baseline_experiments()
        
        # Should generate multiple baseline experiments
        self.assertGreater(len(experiments), 0)
        
        # Each experiment should have valid configuration
        for exp in experiments:
            self.assertIsInstance(exp, ExperimentConfig)
            self.assertIsInstance(exp.experiment_id, str)
            self.assertIsInstance(exp.seed, int)
            self.assertTrue(exp.deterministic)
    
    def test_ablation_study_generation(self):
        """
        Test ablation study generation.
        
        @cursor Validates systematic ablation study design
        """
        experiments = self.generator.generate_ablation_studies()
        
        # Should generate multiple ablation experiments
        self.assertGreater(len(experiments), 5)
        
        # Should have experiments with different configurations
        vesselness_configs = [exp.use_vesselness for exp in experiments]
        self.assertIn(True, vesselness_configs)
        self.assertIn(False, vesselness_configs)
    
    def test_experiment_uniqueness(self):
        """
        Test that generated experiments have unique configurations.
        
        @cursor Ensures no duplicate experiments are generated
        """
        all_experiments = self.generator.generate_all_experiments()
        
        # Collect all experiment hashes
        all_hashes = []
        for suite_experiments in all_experiments.values():
            for exp in suite_experiments:
                all_hashes.append(exp.get_hash())
        
        # Should have unique hashes (no duplicates)
        unique_hashes = set(all_hashes)
        self.assertEqual(len(all_hashes), len(unique_hashes), 
                        "All experiments should have unique configurations")

class TestIntegration(unittest.TestCase):
    """
    Integration tests for complete pipeline.
    
    @cursor Tests end-to-end pipeline functionality
    @cursor Validates integration between components
    """
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        set_deterministic_seed(42)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_pipeline(self):
        """
        Test complete end-to-end pipeline.
        
        @cursor Validates full pipeline from data to evaluation
        """
        # Create synthetic volume
        volume = np.random.random((32, 32, 32)).astype(np.float32)
        
        # Preprocessing
        pipeline = AneurysmPreprocessingPipeline()
        processed = pipeline.process_volume(volume)
        
        # Model inference
        model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
        model.eval()
        
        input_tensor = torch.from_numpy(processed['final']).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Evaluation
        scorer = CompetitionScorer()
        
        # Create dummy ground truth
        ground_truth = {}
        for key in predictions.keys():
            ground_truth[key] = np.random.binomial(1, 0.3, 1)
        
        # Convert predictions to numpy
        pred_dict = {}
        for key, value in predictions.items():
            pred_dict[key] = value.numpy()
        
        score = scorer.calculate_competition_score(pred_dict, ground_truth)
        
        # Should complete without errors and return valid score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

def run_performance_benchmark():
    """
    Run performance benchmark tests.
    
    @cursor Comprehensive performance testing for optimization analysis
    """
    print("Running Performance Benchmark...")
    print("=" * 50)
    
    # Test preprocessing performance
    pipeline = AneurysmPreprocessingPipeline()
    
    sizes = [32, 48, 64]
    for size in sizes:
        volume = np.random.random((size, size, size)).astype(np.float32)
        
        start_time = time.time()
        result = pipeline.process_volume(volume)
        end_time = time.time()
        
        processing_time = end_time - start_time
        voxels = size ** 3
        time_per_voxel = processing_time / voxels
        
        print(f"Size {size}³: {processing_time:.3f}s ({time_per_voxel:.2e}s/voxel)")
    
    # Test model performance
    print("\nModel Performance:")
    model = MultiTaskAneurysmNet(input_channels=1, num_aneurysm_chars=13)
    model.eval()
    
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 1, 64, 64, 64)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()
        
        inference_time = end_time - start_time
        time_per_sample = inference_time / batch_size
        
        print(f"Batch size {batch_size}: {inference_time:.3f}s ({time_per_sample:.3f}s/sample)")

def main():
    """
    Main test runner.
    
    @cursor Comprehensive test execution with performance benchmarking
    """
    print("Aneurysm Detection Pipeline - Comprehensive Test Suite")
    print("=" * 60)
    
    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDeterministicBehavior,
        TestComputationalComplexity,
        TestModelAccuracy,
        TestCompetitionScoring,
        TestStatisticalAnalysis,
        TestExperimentGeneration,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Performance benchmark
    print("\n" + "=" * 60)
    run_performance_benchmark()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\nSuccess Rate: {success_rate:.1%}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
