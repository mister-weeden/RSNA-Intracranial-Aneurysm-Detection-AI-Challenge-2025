import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
import pandas as pd

class AneurysmCompetitionScorer:
    """
    Scoring function for aneurysm detection competition.
    Implements weighted average of AUC scores with emphasis on aneurysm detection.
    """
    
    def __init__(self, vessel_weight: float = 0.1, aneurysm_weight: float = 0.5):
        """
        Initialize scorer with task weights.
        
        Args:
            vessel_weight: Weight for each vessel classification task (default: 0.1)
            aneurysm_weight: Weight for aneurysm detection task (default: 0.5)
        """
        self.vessel_classes = ['ICA', 'MCA', 'ACA', 'Pcom']
        self.vessel_weight = vessel_weight
        self.aneurysm_weight = aneurysm_weight
        
        # Validate weights sum to 1.0
        total_weight = len(self.vessel_classes) * vessel_weight + aneurysm_weight
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        self.scores_history = []
        
    def calculate_auc_ap(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         task_name: str = "task") -> float:
        """
        Calculate Average Precision (AUC-AP) for binary classification.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities
            task_name: Name of the task for logging
            
        Returns:
            Average Precision score
        """
        try:
            # Handle edge cases
            if len(np.unique(y_true)) == 1:
                print(f"Warning: Only one class present in {task_name}. AUC-AP undefined.")
                return 0.5  # Default score for undefined cases
            
            ap_score = average_precision_score(y_true, y_pred)
            return ap_score
        except Exception as e:
            print(f"Error calculating AUC-AP for {task_name}: {e}")
            return 0.0
    
    def calculate_competition_score(self, 
                                   predictions: Dict[str, np.ndarray],
                                   ground_truth: Dict[str, np.ndarray],
                                   verbose: bool = True) -> Dict[str, float]:
        """
        Calculate the competition score based on weighted average of AUCs.
        
        Args:
            predictions: Dictionary with keys ['ICA', 'MCA', 'ACA', 'Pcom', 'aneurysm']
                        containing prediction probabilities
            ground_truth: Dictionary with same keys containing ground truth labels
            verbose: Print detailed scores for each component
            
        Returns:
            Dictionary containing individual scores and final weighted score
        """
        scores = {}
        weighted_sum = 0.0
        
        # Calculate vessel classification scores
        for vessel in self.vessel_classes:
            if vessel not in predictions or vessel not in ground_truth:
                print(f"Warning: {vessel} not found in predictions/ground_truth")
                scores[f'AUC_{vessel}'] = 0.5  # Default score
                continue
                
            auc = self.calculate_auc_ap(ground_truth[vessel], predictions[vessel], vessel)
            scores[f'AUC_{vessel}'] = auc
            weighted_sum += self.vessel_weight * auc
            
            if verbose:
                print(f"AUC_{vessel}: {auc:.4f} (weight: {self.vessel_weight})")
        
        # Calculate aneurysm detection score (main task)
        if 'aneurysm' in predictions and 'aneurysm' in ground_truth:
            aneurysm_auc = self.calculate_auc_ap(
                ground_truth['aneurysm'], 
                predictions['aneurysm'], 
                'aneurysm'
            )
            scores['AUC_aneurysm'] = aneurysm_auc
            weighted_sum += self.aneurysm_weight * aneurysm_auc
            
            if verbose:
                print(f"AUC_aneurysm: {aneurysm_auc:.4f} (weight: {self.aneurysm_weight})")
        else:
            print("Error: Aneurysm predictions not found!")
            scores['AUC_aneurysm'] = 0.0
        
        # Calculate final score
        scores['weighted_score'] = weighted_sum
        
        if verbose:
            print(f"\nFinal Weighted Score: {weighted_sum:.4f}")
            print(f"Contribution from vessels: {sum([scores[f'AUC_{v}'] * self.vessel_weight for v in self.vessel_classes]):.4f}")
            print(f"Contribution from aneurysm: {scores['AUC_aneurysm'] * self.aneurysm_weight:.4f}")
        
        # Store in history
        self.scores_history.append(scores)
        
        return scores
    
    def validate_minimum_requirements(self, scores: Dict[str, float]) -> bool:
        """
        Check if scores meet minimum requirements (no vessel AUC < 0.6).
        
        Args:
            scores: Dictionary of calculated scores
            
        Returns:
            True if all requirements are met
        """
        min_vessel_auc = 0.6
        
        for vessel in self.vessel_classes:
            vessel_score = scores.get(f'AUC_{vessel}', 0.0)
            if vessel_score < min_vessel_auc:
                print(f"Warning: {vessel} AUC ({vessel_score:.4f}) below minimum threshold ({min_vessel_auc})")
                return False
        
        return True
    
    def simulate_scores(self, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Simulate different score scenarios to understand scoring dynamics.
        
        Args:
            n_samples: Number of samples to simulate
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with simulated scenarios
        """
        np.random.seed(seed)
        scenarios = []
        
        for i in range(n_samples):
            # Simulate realistic AUC ranges
            vessel_aucs = np.random.beta(8, 3, size=4)  # Tends toward 0.7-0.8
            aneurysm_auc = np.random.beta(6, 4, size=1)[0]  # Tends toward 0.6-0.7
            
            # Create predictions and ground truth
            predictions = {
                'ICA': vessel_aucs[0],
                'MCA': vessel_aucs[1],
                'ACA': vessel_aucs[2],
                'Pcom': vessel_aucs[3],
                'aneurysm': aneurysm_auc
            }
            
            # Calculate weighted score
            weighted = sum(vessel_aucs) * self.vessel_weight + aneurysm_auc * self.aneurysm_weight
            
            scenarios.append({
                'scenario_id': i,
                'AUC_ICA': vessel_aucs[0],
                'AUC_MCA': vessel_aucs[1],
                'AUC_ACA': vessel_aucs[2],
                'AUC_Pcom': vessel_aucs[3],
                'AUC_aneurysm': aneurysm_auc,
                'weighted_score': weighted,
                'avg_vessel_auc': np.mean(vessel_aucs),
                'min_vessel_auc': np.min(vessel_aucs),
                'vessels_contribution': sum(vessel_aucs) * self.vessel_weight,
                'aneurysm_contribution': aneurysm_auc * self.aneurysm_weight
            })
        
        df = pd.DataFrame(scenarios)
        
        # Add strategic insights
        df['strategy_optimal'] = (df['AUC_aneurysm'] > 0.7) & (df['min_vessel_auc'] > 0.6)
        df['aneurysm_focused'] = df['aneurysm_contribution'] > df['vessels_contribution']
        
        return df
    
    def plot_score_analysis(self, simulated_df: Optional[pd.DataFrame] = None):
        """
        Create visualization of score distributions and strategies.
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
            
            if simulated_df is None:
                simulated_df = self.simulate_scores()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Score distribution
            ax1 = axes[0, 0]
            ax1.hist(simulated_df['weighted_score'], bins=30, alpha=0.7, color='blue')
            ax1.axvline(simulated_df['weighted_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {simulated_df["weighted_score"].mean():.3f}')
            ax1.set_xlabel('Weighted Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Competition Scores')
            ax1.legend()
            
            # Plot 2: Contribution analysis
            ax2 = axes[0, 1]
            ax2.scatter(simulated_df['aneurysm_contribution'], 
                       simulated_df['vessels_contribution'],
                       c=simulated_df['weighted_score'], cmap='viridis', alpha=0.6)
            ax2.set_xlabel('Aneurysm Contribution')
            ax2.set_ylabel('Vessels Contribution')
            ax2.set_title('Task Contributions to Final Score')
            plt.colorbar(ax2.collections[0], ax=ax2, label='Weighted Score')
            
            # Plot 3: Strategy comparison
            ax3 = axes[1, 0]
            optimal = simulated_df[simulated_df['strategy_optimal']]
            suboptimal = simulated_df[~simulated_df['strategy_optimal']]
            
            ax3.hist([optimal['weighted_score'], suboptimal['weighted_score']], 
                    label=['Optimal Strategy', 'Suboptimal'], 
                    bins=20, alpha=0.7, color=['green', 'red'])
            ax3.set_xlabel('Weighted Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Score Distribution by Strategy')
            ax3.legend()
            
            # Plot 4: Aneurysm AUC vs Final Score
            ax4 = axes[1, 1]
            ax4.scatter(simulated_df['AUC_aneurysm'], simulated_df['weighted_score'], 
                       alpha=0.5, s=10)
            z = np.polyfit(simulated_df['AUC_aneurysm'], simulated_df['weighted_score'], 1)
            p = np.poly1d(z)
            ax4.plot(simulated_df['AUC_aneurysm'].sort_values(), 
                    p(simulated_df['AUC_aneurysm'].sort_values()),
                    "r--", alpha=0.8, label=f'Correlation: {simulated_df["AUC_aneurysm"].corr(simulated_df["weighted_score"]):.3f}')
            ax4.set_xlabel('Aneurysm AUC')
            ax4.set_ylabel('Final Weighted Score')
            ax4.set_title('Impact of Aneurysm Detection on Final Score')
            ax4.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
    
    def get_strategy_recommendations(self, current_scores: Optional[Dict[str, float]] = None) -> str:
        """
        Provide strategic recommendations based on current or simulated scores.
        
        Args:
            current_scores: Current model scores (if available)
            
        Returns:
            String with strategic recommendations
        """
        recommendations = []
        
        if current_scores:
            aneurysm_score = current_scores.get('AUC_aneurysm', 0)
            vessel_scores = [current_scores.get(f'AUC_{v}', 0) for v in self.vessel_classes]
            min_vessel = min(vessel_scores)
            avg_vessel = np.mean(vessel_scores)
            
            recommendations.append(f"Current Performance Analysis:")
            recommendations.append(f"- Aneurysm AUC: {aneurysm_score:.4f}")
            recommendations.append(f"- Average Vessel AUC: {avg_vessel:.4f}")
            recommendations.append(f"- Minimum Vessel AUC: {min_vessel:.4f}")
            recommendations.append("")
            
            # Priority recommendations
            if aneurysm_score < 0.7:
                recommendations.append("🔴 HIGH PRIORITY: Focus on aneurysm detection")
                recommendations.append("   - This contributes 50% of your final score")
                recommendations.append("   - Consider: Frangi filtering, synthetic augmentation, focal loss")
            
            if min_vessel < 0.6:
                recommendations.append("🟡 MEDIUM PRIORITY: Improve worst-performing vessel")
                vessel_idx = vessel_scores.index(min_vessel)
                recommendations.append(f"   - {self.vessel_classes[vessel_idx]} needs attention")
                recommendations.append("   - Risk of dragging down overall score")
            
            if aneurysm_score > 0.75 and avg_vessel < 0.7:
                recommendations.append("🟢 OPTIMIZATION: Vessel classifications need improvement")
                recommendations.append("   - Aneurysm detection is strong, focus on vessels")
                recommendations.append("   - Consider multi-task learning to share features")
        
        else:
            recommendations.append("General Strategic Recommendations:")
            recommendations.append("")
            recommendations.append("1. ANEURYSM DETECTION (70% effort):")
            recommendations.append("   - Implement Frangi/Hessian vessel enhancement")
            recommendations.append("   - Use temporal MIP for better visualization")
            recommendations.append("   - Apply synthetic aneurysm augmentation")
            recommendations.append("   - Consider focal loss for class imbalance")
            recommendations.append("")
            recommendations.append("2. VESSEL CLASSIFICATION (30% effort):")
            recommendations.append("   - Ensure no vessel AUC < 0.6")
            recommendations.append("   - Use multi-task learning architecture")
            recommendations.append("   - Share backbone features with aneurysm detection")
            recommendations.append("")
            recommendations.append("3. PREPROCESSING PIPELINE:")
            recommendations.append("   - Frangi filtering (benefits both tasks)")
            recommendations.append("   - Centerline alignment")
            recommendations.append("   - Intensity normalization")
            recommendations.append("   - Aneurysm-specific augmentations")
        
        return "\n".join(recommendations)


# Example usage and validation
if __name__ == "__main__":
    # Initialize scorer
    scorer = AneurysmCompetitionScorer(vessel_weight=0.1, aneurysm_weight=0.5)
    
    # Example 1: Calculate score with dummy data
    print("=" * 60)
    print("EXAMPLE 1: Basic Scoring")
    print("=" * 60)
    
    # Generate dummy predictions and ground truth
    np.random.seed(42)
    n_samples = 1000
    
    predictions = {
        'ICA': np.random.rand(n_samples),
        'MCA': np.random.rand(n_samples),
        'ACA': np.random.rand(n_samples),
        'Pcom': np.random.rand(n_samples),
        'aneurysm': np.random.rand(n_samples)
    }
    
    ground_truth = {
        'ICA': np.random.randint(0, 2, n_samples),
        'MCA': np.random.randint(0, 2, n_samples),
        'ACA': np.random.randint(0, 2, n_samples),
        'Pcom': np.random.randint(0, 2, n_samples),
        'aneurysm': np.random.randint(0, 2, n_samples)
    }
    
    scores = scorer.calculate_competition_score(predictions, ground_truth)
    
    # Check minimum requirements
    print(f"\nMeets minimum requirements: {scorer.validate_minimum_requirements(scores)}")
    
    # Example 2: Simulate scoring scenarios
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Score Simulation Analysis")
    print("=" * 60)
    
    simulated_df = scorer.simulate_scores(n_samples=5000)
    
    print(f"\nSimulation Statistics:")
    print(f"Mean weighted score: {simulated_df['weighted_score'].mean():.4f}")
    print(f"Std weighted score: {simulated_df['weighted_score'].std():.4f}")
    print(f"Optimal strategy scenarios: {simulated_df['strategy_optimal'].sum()} / {len(simulated_df)}")
    print(f"Aneurysm-focused scenarios: {simulated_df['aneurysm_focused'].sum()} / {len(simulated_df)}")
    
    # Example 3: Strategic recommendations
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Strategic Recommendations")
    print("=" * 60)
    
    # With current scores
    current_scores = {
        'AUC_ICA': 0.72,
        'AUC_MCA': 0.68,
        'AUC_ACA': 0.58,  # Below threshold
        'AUC_Pcom': 0.71,
        'AUC_aneurysm': 0.65  # Needs improvement
    }
    
    print("\n" + scorer.get_strategy_recommendations(current_scores))
    
    # Example 4: Visualization (if matplotlib available)
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Generating Visualizations")
    print("=" * 60)
    
    # Uncomment to generate plots
    scorer.plot_score_analysis(simulated_df)
    
    print("\nScoring function ready for competition use!")