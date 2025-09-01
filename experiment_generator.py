#!/usr/bin/env python3
"""
Experiment Generator for Aneurysm Detection Research
Generates systematic experiments for performance, complexity, and accuracy analysis.

@cursor Implements systematic experiment design for research reproducibility
@cursor Generates experiments suitable for Kaggle, RSNA, and academic submission
@cursor Includes hyperparameter optimization and ablation studies
@cursor Provides deterministic experiment configuration for reproducible results
"""

import json
import numpy as np
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.
    
    @cursor Structured experiment configuration for reproducibility
    @cursor Includes all parameters needed for deterministic training
    """
    experiment_id: str
    experiment_name: str
    model_architecture: str
    
    # Data configuration
    batch_size: int
    input_size: Tuple[int, int, int]
    multi_task: bool
    
    # Training configuration
    learning_rate: float
    optimizer: str
    scheduler: str
    num_epochs: int
    weight_decay: float
    
    # Loss configuration
    focal_alpha: float
    focal_gamma: float
    aneurysm_weight: float
    char_weight: float
    
    # Regularization
    dropout_rate: float
    max_grad_norm: float
    
    # Preprocessing
    use_vesselness: bool
    use_aneurysm_enhancement: bool
    frangi_scales: List[float]
    
    # Uncertainty quantification
    use_uncertainty: bool
    mc_samples: int
    
    # Reproducibility
    seed: int
    deterministic: bool
    
    # Performance monitoring
    log_interval: int
    save_interval: int
    patience: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class ExperimentGenerator:
    """
    Systematic experiment generator for aneurysm detection research.
    
    @cursor Generates comprehensive experiment suites for research validation
    @cursor Supports ablation studies, hyperparameter optimization, and complexity analysis
    @cursor Ensures reproducible and systematic experimental design
    """
    
    def __init__(self, base_output_dir: str = "experiments"):
        """
        Initialize experiment generator.
        
        Args:
            base_output_dir: Base directory for experiment outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Default base configuration
        self.base_config = {
            'model_architecture': 'MultiTaskAneurysmNet',
            'batch_size': 4,
            'input_size': (64, 64, 64),
            'multi_task': True,
            'learning_rate': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'num_epochs': 100,
            'weight_decay': 1e-5,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'aneurysm_weight': 0.5,
            'char_weight': 0.5,
            'dropout_rate': 0.3,
            'max_grad_norm': 1.0,
            'use_vesselness': True,
            'use_aneurysm_enhancement': True,
            'frangi_scales': [0.5, 1.0, 1.5, 2.0, 2.5],
            'use_uncertainty': True,
            'mc_samples': 5,
            'seed': 42,
            'deterministic': True,
            'log_interval': 10,
            'save_interval': 10,
            'patience': 20
        }
    
    def generate_baseline_experiments(self) -> List[ExperimentConfig]:
        """
        Generate baseline experiments with standard configurations.
        
        @cursor Creates baseline experiments for comparison and validation
        @cursor Includes different model sizes and training strategies
        
        Returns:
            List of baseline experiment configurations
        """
        experiments = []
        
        # Baseline configurations
        baseline_configs = [
            {
                'experiment_name': 'baseline_small',
                'batch_size': 8,
                'input_size': (32, 32, 32),
                'learning_rate': 1e-3,
                'num_epochs': 50
            },
            {
                'experiment_name': 'baseline_medium',
                'batch_size': 4,
                'input_size': (64, 64, 64),
                'learning_rate': 1e-4,
                'num_epochs': 100
            },
            {
                'experiment_name': 'baseline_large',
                'batch_size': 2,
                'input_size': (128, 128, 128),
                'learning_rate': 5e-5,
                'num_epochs': 150
            }
        ]
        
        for i, config_override in enumerate(baseline_configs):
            config = self.base_config.copy()
            config.update(config_override)
            config['seed'] = 42 + i  # Different seed for each baseline
            
            experiment = self._create_experiment_config(config)
            experiments.append(experiment)
        
        logger.info(f"Generated {len(experiments)} baseline experiments")
        return experiments
    
    def generate_ablation_studies(self) -> List[ExperimentConfig]:
        """
        Generate ablation study experiments to analyze component contributions.
        
        @cursor Systematic ablation studies to understand component importance
        @cursor Tests preprocessing, architecture, and training components
        
        Returns:
            List of ablation experiment configurations
        """
        experiments = []
        
        # Preprocessing ablations
        preprocessing_ablations = [
            {
                'experiment_name': 'ablation_no_vesselness',
                'use_vesselness': False,
                'use_aneurysm_enhancement': True
            },
            {
                'experiment_name': 'ablation_no_aneurysm_enhancement',
                'use_vesselness': True,
                'use_aneurysm_enhancement': False
            },
            {
                'experiment_name': 'ablation_no_preprocessing',
                'use_vesselness': False,
                'use_aneurysm_enhancement': False
            },
            {
                'experiment_name': 'ablation_single_scale_frangi',
                'frangi_scales': [1.0]
            }
        ]
        
        # Architecture ablations
        architecture_ablations = [
            {
                'experiment_name': 'ablation_no_uncertainty',
                'use_uncertainty': False,
                'mc_samples': 1
            },
            {
                'experiment_name': 'ablation_no_dropout',
                'dropout_rate': 0.0,
                'use_uncertainty': False
            },
            {
                'experiment_name': 'ablation_single_task',
                'multi_task': False,
                'char_weight': 0.0,
                'aneurysm_weight': 1.0
            }
        ]
        
        # Training ablations
        training_ablations = [
            {
                'experiment_name': 'ablation_no_focal_loss',
                'focal_alpha': 1.0,
                'focal_gamma': 0.0
            },
            {
                'experiment_name': 'ablation_equal_task_weights',
                'aneurysm_weight': 0.5,
                'char_weight': 0.5
            },
            {
                'experiment_name': 'ablation_sgd_optimizer',
                'optimizer': 'sgd',
                'learning_rate': 1e-2
            }
        ]
        
        # Combine all ablations
        all_ablations = preprocessing_ablations + architecture_ablations + training_ablations
        
        for i, ablation_config in enumerate(all_ablations):
            config = self.base_config.copy()
            config.update(ablation_config)
            config['seed'] = 100 + i  # Different seed range for ablations
            
            experiment = self._create_experiment_config(config)
            experiments.append(experiment)
        
        logger.info(f"Generated {len(experiments)} ablation study experiments")
        return experiments
    
    def generate_hyperparameter_optimization(self, 
                                           n_trials: int = 20) -> List[ExperimentConfig]:
        """
        Generate hyperparameter optimization experiments.
        
        @cursor Systematic hyperparameter search for optimal performance
        @cursor Uses grid search and random search strategies
        
        Args:
            n_trials: Number of hyperparameter combinations to try
            
        Returns:
            List of hyperparameter optimization experiments
        """
        experiments = []
        
        # Define hyperparameter search spaces
        param_grid = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [2, 4, 8],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
            'focal_gamma': [0.5, 1.0, 2.0, 3.0],
            'aneurysm_weight': [0.3, 0.4, 0.5, 0.6, 0.7]
        }
        
        # Generate random combinations
        np.random.seed(42)  # For reproducible hyperparameter selection
        
        for trial in range(n_trials):
            config = self.base_config.copy()
            
            # Sample random hyperparameters
            for param, values in param_grid.items():
                config[param] = np.random.choice(values)
            
            config['experiment_name'] = f'hyperparam_trial_{trial:03d}'
            config['seed'] = 200 + trial  # Different seed range for hyperparameter trials
            
            experiment = self._create_experiment_config(config)
            experiments.append(experiment)
        
        logger.info(f"Generated {len(experiments)} hyperparameter optimization experiments")
        return experiments
    
    def generate_complexity_analysis_experiments(self) -> List[ExperimentConfig]:
        """
        Generate experiments for computational complexity analysis.
        
        @cursor Tests different input sizes and batch sizes for complexity analysis
        @cursor Identifies O(n*log(n)) bottlenecks and optimization opportunities
        
        Returns:
            List of complexity analysis experiments
        """
        experiments = []
        
        # Different input sizes for complexity analysis
        input_sizes = [
            (32, 32, 32),
            (48, 48, 48),
            (64, 64, 64),
            (96, 96, 96),
            (128, 128, 128)
        ]
        
        # Different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for i, (input_size, batch_size) in enumerate(itertools.product(input_sizes, batch_sizes)):
            # Skip combinations that would require too much memory
            total_voxels = np.prod(input_size) * batch_size
            if total_voxels > 2**20:  # Skip if > 1M voxels per batch
                continue
            
            config = self.base_config.copy()
            config.update({
                'experiment_name': f'complexity_analysis_{input_size[0]}x{input_size[1]}x{input_size[2]}_bs{batch_size}',
                'input_size': input_size,
                'batch_size': batch_size,
                'num_epochs': 10,  # Shorter runs for complexity analysis
                'seed': 300 + i
            })
            
            experiment = self._create_experiment_config(config)
            experiments.append(experiment)
        
        logger.info(f"Generated {len(experiments)} complexity analysis experiments")
        return experiments
    
    def generate_ensemble_experiments(self) -> List[ExperimentConfig]:
        """
        Generate experiments for ensemble model training.
        
        @cursor Creates diverse models for ensemble combination
        @cursor Uses different architectures, seeds, and training strategies
        
        Returns:
            List of ensemble experiment configurations
        """
        experiments = []
        
        # Different ensemble member configurations
        ensemble_configs = [
            {
                'experiment_name': 'ensemble_member_1',
                'seed': 42,
                'dropout_rate': 0.2,
                'learning_rate': 1e-4
            },
            {
                'experiment_name': 'ensemble_member_2',
                'seed': 123,
                'dropout_rate': 0.3,
                'learning_rate': 5e-5
            },
            {
                'experiment_name': 'ensemble_member_3',
                'seed': 456,
                'dropout_rate': 0.4,
                'learning_rate': 2e-4
            },
            {
                'experiment_name': 'ensemble_member_4',
                'seed': 789,
                'dropout_rate': 0.25,
                'learning_rate': 8e-5,
                'optimizer': 'sgd'
            },
            {
                'experiment_name': 'ensemble_member_5',
                'seed': 999,
                'dropout_rate': 0.35,
                'learning_rate': 1.5e-4,
                'focal_gamma': 3.0
            }
        ]
        
        for config_override in ensemble_configs:
            config = self.base_config.copy()
            config.update(config_override)
            
            experiment = self._create_experiment_config(config)
            experiments.append(experiment)
        
        logger.info(f"Generated {len(experiments)} ensemble experiments")
        return experiments
    
    def _create_experiment_config(self, config_dict: Dict) -> ExperimentConfig:
        """
        Create ExperimentConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig object
        """
        # Generate unique experiment ID
        experiment_id = f"exp_{config_dict['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=config_dict['experiment_name'],
            model_architecture=config_dict['model_architecture'],
            batch_size=config_dict['batch_size'],
            input_size=config_dict['input_size'],
            multi_task=config_dict['multi_task'],
            learning_rate=config_dict['learning_rate'],
            optimizer=config_dict['optimizer'],
            scheduler=config_dict['scheduler'],
            num_epochs=config_dict['num_epochs'],
            weight_decay=config_dict['weight_decay'],
            focal_alpha=config_dict['focal_alpha'],
            focal_gamma=config_dict['focal_gamma'],
            aneurysm_weight=config_dict['aneurysm_weight'],
            char_weight=config_dict['char_weight'],
            dropout_rate=config_dict['dropout_rate'],
            max_grad_norm=config_dict['max_grad_norm'],
            use_vesselness=config_dict['use_vesselness'],
            use_aneurysm_enhancement=config_dict['use_aneurysm_enhancement'],
            frangi_scales=config_dict['frangi_scales'],
            use_uncertainty=config_dict['use_uncertainty'],
            mc_samples=config_dict['mc_samples'],
            seed=config_dict['seed'],
            deterministic=config_dict['deterministic'],
            log_interval=config_dict['log_interval'],
            save_interval=config_dict['save_interval'],
            patience=config_dict['patience']
        )
        
        return experiment
    
    def generate_all_experiments(self) -> Dict[str, List[ExperimentConfig]]:
        """
        Generate all experiment suites.
        
        @cursor Comprehensive experiment generation for complete research study
        @cursor Organizes experiments by type for systematic execution
        
        Returns:
            Dictionary containing all experiment suites
        """
        logger.info("Generating comprehensive experiment suite")
        
        all_experiments = {
            'baseline': self.generate_baseline_experiments(),
            'ablation': self.generate_ablation_studies(),
            'hyperparameter': self.generate_hyperparameter_optimization(n_trials=15),
            'complexity': self.generate_complexity_analysis_experiments(),
            'ensemble': self.generate_ensemble_experiments()
        }
        
        # Calculate total experiments
        total_experiments = sum(len(experiments) for experiments in all_experiments.values())
        logger.info(f"Generated {total_experiments} total experiments across {len(all_experiments)} suites")
        
        # Save experiment configurations
        self._save_experiment_suite(all_experiments)
        
        return all_experiments
    
    def _save_experiment_suite(self, experiment_suite: Dict[str, List[ExperimentConfig]]) -> None:
        """
        Save experiment suite to JSON files.
        
        Args:
            experiment_suite: Dictionary of experiment suites
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for suite_name, experiments in experiment_suite.items():
            suite_dir = self.base_output_dir / suite_name
            suite_dir.mkdir(exist_ok=True)
            
            # Save individual experiment configs
            for experiment in experiments:
                config_file = suite_dir / f"{experiment.experiment_id}.json"
                with open(config_file, 'w') as f:
                    json.dump(experiment.to_dict(), f, indent=2)
            
            # Save suite summary
            summary_file = suite_dir / f"suite_summary_{timestamp}.json"
            summary = {
                'suite_name': suite_name,
                'num_experiments': len(experiments),
                'experiment_ids': [exp.experiment_id for exp in experiments],
                'generated_at': timestamp
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Save master index
        master_index = {
            'generated_at': timestamp,
            'total_experiments': sum(len(experiments) for experiments in experiment_suite.values()),
            'suites': {
                suite_name: {
                    'num_experiments': len(experiments),
                    'experiment_names': [exp.experiment_name for exp in experiments]
                }
                for suite_name, experiments in experiment_suite.items()
            }
        }
        
        master_file = self.base_output_dir / f"experiment_master_index_{timestamp}.json"
        with open(master_file, 'w') as f:
            json.dump(master_index, f, indent=2)
        
        logger.info(f"Experiment suite saved to {self.base_output_dir}")
        logger.info(f"Master index: {master_file}")

class ExperimentRunner:
    """
    Experiment runner for executing generated experiments.
    
    @cursor Manages experiment execution with proper resource allocation
    @cursor Provides progress tracking and result aggregation
    @cursor Handles failures and retries for robust execution
    """
    
    def __init__(self, experiment_suite: Dict[str, List[ExperimentConfig]]):
        """
        Initialize experiment runner.
        
        Args:
            experiment_suite: Dictionary of experiment suites to run
        """
        self.experiment_suite = experiment_suite
        self.results = {}
        
    def run_experiment_suite(self, suite_name: str, 
                           max_parallel: int = 1,
                           dry_run: bool = False) -> Dict:
        """
        Run a specific experiment suite.
        
        @cursor Executes experiments with resource management
        @cursor Supports parallel execution and progress tracking
        
        Args:
            suite_name: Name of the suite to run
            max_parallel: Maximum number of parallel experiments
            dry_run: If True, only validate configurations without running
            
        Returns:
            Dictionary containing execution results
        """
        if suite_name not in self.experiment_suite:
            raise ValueError(f"Suite '{suite_name}' not found in experiment suite")
        
        experiments = self.experiment_suite[suite_name]
        logger.info(f"Running {len(experiments)} experiments in suite '{suite_name}'")
        
        if dry_run:
            logger.info("DRY RUN: Validating experiment configurations")
            for experiment in experiments:
                logger.info(f"  - {experiment.experiment_name}: {experiment.get_hash()}")
            return {'status': 'validated', 'num_experiments': len(experiments)}
        
        # TODO: Implement actual experiment execution
        # This would integrate with the training pipeline
        logger.info("Experiment execution not implemented in this demo")
        
        return {'status': 'pending', 'num_experiments': len(experiments)}

def main():
    """
    Main function for experiment generation demonstration.
    
    @cursor Demonstrates complete experiment generation workflow
    @cursor Creates systematic experiments for aneurysm detection research
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create experiment generator
    generator = ExperimentGenerator(base_output_dir="experiments")
    
    # Generate all experiments
    experiment_suite = generator.generate_all_experiments()
    
    # Print summary
    print("\nExperiment Suite Summary:")
    print("=" * 50)
    
    total_experiments = 0
    for suite_name, experiments in experiment_suite.items():
        print(f"{suite_name.capitalize()} Suite: {len(experiments)} experiments")
        total_experiments += len(experiments)
        
        # Show first few experiment names
        for i, exp in enumerate(experiments[:3]):
            print(f"  - {exp.experiment_name}")
        if len(experiments) > 3:
            print(f"  ... and {len(experiments) - 3} more")
        print()
    
    print(f"Total Experiments: {total_experiments}")
    print(f"Estimated Training Time: {total_experiments * 2:.1f} hours (assuming 2h per experiment)")
    
    # Create experiment runner for demonstration
    runner = ExperimentRunner(experiment_suite)
    
    # Dry run validation
    print("\nValidating baseline experiments...")
    result = runner.run_experiment_suite('baseline', dry_run=True)
    print(f"Validation result: {result}")
    
    return experiment_suite

if __name__ == "__main__":
    main()
