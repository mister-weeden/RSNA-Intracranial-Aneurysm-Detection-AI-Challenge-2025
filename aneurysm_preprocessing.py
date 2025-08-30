import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter, median_filter
from skimage import filters, morphology, exposure
from typing import Tuple, Optional, Dict, List
import warnings

class AneurysmPreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline optimized for aneurysm detection
    with vessel enhancement capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessing pipeline with configuration.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or self.get_default_config()
        
    @staticmethod
    def get_default_config() -> Dict:
        """Get default preprocessing configuration."""
        return {
            'frangi': {
                'scales': [0.5, 1.0, 1.5, 2.0, 2.5],  # Multi-scale for different vessel sizes
                'alpha': 0.5,  # Plate-like structure suppression
                'beta': 0.5,   # Blob-like structure suppression
                'gamma': 15,   # Background suppression
            },
            'intensity': {
                'clip_percentile': (0.5, 99.5),
                'normalize_range': (0, 1),
            },
            'aneurysm': {
                'min_size': 3,  # Minimum aneurysm size in voxels
                'max_size': 30, # Maximum aneurysm size in voxels
                'enhancement_factor': 1.5,
            },
            'temporal': {
                'use_mip': True,
                'mip_slices': 3,  # Number of slices for MIP
            },
            'augmentation': {
                'synthetic_aneurysm_prob': 0.3,
                'synthetic_size_range': (3, 15),
            }
        }
    
    def frangi_vesselness_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply 3D Frangi vesselness filter for vessel enhancement.
        
        Args:
            volume: 3D medical image volume
            
        Returns:
            Vesselness enhanced volume
        """
        scales = self.config['frangi']['scales']
        alpha = self.config['frangi']['alpha']
        beta = self.config['frangi']['beta']
        gamma = self.config['frangi']['gamma']
        
        # Initialize output
        vesselness = np.zeros_like(volume, dtype=np.float32)
        
        for scale in scales:
            # Compute Hessian matrix at current scale
            hessian = self._compute_hessian_3d(volume, sigma=scale)
            
            # Compute eigenvalues
            eigenvalues = self._compute_eigenvalues_3d(hessian)
            
            # Compute vesselness measure
            v_scale = self._frangi_measure_3d(eigenvalues, alpha, beta, gamma)
            
            # Take maximum response across scales
            vesselness = np.maximum(vesselness, v_scale)
        
        return vesselness
    
    def _compute_hessian_3d(self, volume: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute 3D Hessian matrix for each voxel.
        
        Args:
            volume: Input volume
            sigma: Scale parameter
            
        Returns:
            Hessian matrix components
        """
        # Smooth the volume
        smoothed = gaussian_filter(volume, sigma=sigma)
        
        # Compute second derivatives
        hessian = np.zeros(volume.shape + (3, 3), dtype=np.float32)
        
        # Compute gradients
        gradients = np.gradient(smoothed)
        
        # Fill Hessian matrix
        for i in range(3):
            for j in range(3):
                hessian[..., i, j] = np.gradient(gradients[i], axis=j)
        
        return hessian
    
    def _compute_eigenvalues_3d(self, hessian: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues of 3D Hessian matrix.
        
        Args:
            hessian: Hessian matrix for each voxel
            
        Returns:
            Sorted eigenvalues (lambda1 <= lambda2 <= lambda3)
        """
        shape = hessian.shape[:-2]
        eigenvalues = np.zeros(shape + (3,), dtype=np.float32)
        
        # Flatten for efficient computation
        hessian_flat = hessian.reshape(-1, 3, 3)
        
        for i in range(hessian_flat.shape[0]):
            try:
                eigvals = np.linalg.eigvalsh(hessian_flat[i])
                eigenvalues.flat[i*3:(i+1)*3] = np.sort(eigvals)
            except:
                # Handle numerical issues
                eigenvalues.flat[i*3:(i+1)*3] = [0, 0, 0]
        
        return eigenvalues.reshape(shape + (3,))
    
    def _frangi_measure_3d(self, eigenvalues: np.ndarray, 
                          alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Compute Frangi vesselness measure from eigenvalues.
        
        Args:
            eigenvalues: Sorted eigenvalues for each voxel
            alpha, beta, gamma: Frangi filter parameters
            
        Returns:
            Vesselness measure
        """
        lambda1 = eigenvalues[..., 0]
        lambda2 = eigenvalues[..., 1]
        lambda3 = eigenvalues[..., 2]
        
        # Compute ratios
        Ra = np.abs(lambda2) / (np.abs(lambda3) + 1e-10)
        Rb = np.abs(lambda1) / (np.sqrt(np.abs(lambda2 * lambda3)) + 1e-10)
        S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
        
        # Compute vesselness
        vesselness = np.zeros_like(lambda1)
        
        # Only compute for bright vessels (lambda2, lambda3 < 0)
        mask = (lambda2 < 0) & (lambda3 < 0)
        
        vesselness[mask] = (
            (1 - np.exp(-(Ra[mask]**2) / (2 * alpha**2))) *
            np.exp(-(Rb[mask]**2) / (2 * beta**2)) *
            (1 - np.exp(-(S[mask]**2) / (2 * gamma**2)))
        )
        
        return vesselness
    
    def enhance_aneurysm_regions(self, volume: np.ndarray, 
                                vesselness: np.ndarray) -> np.ndarray:
        """
        Specifically enhance potential aneurysm regions.
        
        Args:
            volume: Original volume
            vesselness: Vessel-enhanced volume
            
        Returns:
            Aneurysm-enhanced volume
        """
        enhanced = volume.copy()
        
        # Detect blob-like structures (potential aneurysms)
        blob_filter = self._detect_blob_structures(volume)
        
        # Find regions with high intensity and blob-like characteristics
        threshold = np.percentile(volume[volume > 0], 90)
        potential_aneurysms = (blob_filter > 0.3) & (volume > threshold)
        
        # Enhance these regions
        enhancement_factor = self.config['aneurysm']['enhancement_factor']
        enhanced[potential_aneurysms] *= enhancement_factor
        
        # Combine with vesselness for connected regions
        vessel_mask = vesselness > np.percentile(vesselness, 75)
        
        # Dilate potential aneurysm regions to connect with vessels
        dilated_aneurysms = morphology.binary_dilation(
            potential_aneurysms, 
            morphology.ball(2)
        )
        
        # Final enhancement
        enhanced[dilated_aneurysms & vessel_mask] *= 1.2
        
        return enhanced
    
    def _detect_blob_structures(self, volume: np.ndarray) -> np.ndarray:
        """
        Detect blob-like structures that could be aneurysms.
        
        Args:
            volume: Input volume
            
        Returns:
            Blob response map
        """
        # Use Laplacian of Gaussian for blob detection
        blob_response = np.zeros_like(volume, dtype=np.float32)
        
        # Multi-scale blob detection
        for sigma in [1.0, 1.5, 2.0, 2.5]:
            # Apply LoG filter
            log_filtered = ndi.gaussian_laplace(volume, sigma=sigma)
            
            # Normalize by scale
            log_filtered *= sigma**2
            
            # Take maximum response
            blob_response = np.maximum(blob_response, -log_filtered)
        
        # Normalize to [0, 1]
        if blob_response.max() > 0:
            blob_response = (blob_response - blob_response.min()) / (blob_response.max() - blob_response.min())
        
        return blob_response
    
    def temporal_mip(self, volume: np.ndarray, axis: int = 2) -> np.ndarray:
        """
        Create Maximum Intensity Projection for better visualization.
        
        Args:
            volume: 3D volume
            axis: Axis along which to compute MIP
            
        Returns:
            MIP-enhanced volume
        """
        if not self.config['temporal']['use_mip']:
            return volume
        
        num_slices = self.config['temporal']['mip_slices']
        enhanced = volume.copy()
        
        # Apply MIP in sliding window fashion
        for i in range(volume.shape[axis]):
            start = max(0, i - num_slices // 2)
            end = min(volume.shape[axis], i + num_slices // 2 + 1)
            
            if axis == 0:
                window = volume[start:end, :, :]
                enhanced[i, :, :] = np.max(window, axis=0)
            elif axis == 1:
                window = volume[:, start:end, :]
                enhanced[:, i, :] = np.max(window, axis=1)
            else:
                window = volume[:, :, start:end]
                enhanced[:, :, i] = np.max(window, axis=2)
        
        return enhanced
    
    def intensity_normalization(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize intensity values with outlier clipping.
        
        Args:
            volume: Input volume
            
        Returns:
            Normalized volume
        """
        # Get clipping percentiles
        p_low, p_high = self.config['intensity']['clip_percentile']
        
        # Compute percentiles on non-zero values
        non_zero_mask = volume > 0
        if non_zero_mask.any():
            v_min = np.percentile(volume[non_zero_mask], p_low)
            v_max = np.percentile(volume[non_zero_mask], p_high)
        else:
            v_min, v_max = volume.min(), volume.max()
        
        # Clip and normalize
        normalized = np.clip(volume, v_min, v_max)
        
        if v_max > v_min:
            normalized = (normalized - v_min) / (v_max - v_min)
        
        # Apply histogram equalization for better contrast
        normalized = exposure.equalize_adapthist(
            normalized, 
            clip_limit=0.03
        )
        
        return normalized
    
    def add_synthetic_aneurysm(self, volume: np.ndarray, 
                               vesselness: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Add synthetic aneurysm for augmentation.
        
        Args:
            volume: Input volume
            vesselness: Vessel map for placement guidance
            
        Returns:
            Augmented volume and augmentation info
        """
        if np.random.random() > self.config['augmentation']['synthetic_aneurysm_prob']:
            return volume, {'added': False}
        
        augmented = volume.copy()
        
        # Find suitable location (high vesselness region)
        vessel_mask = vesselness > np.percentile(vesselness, 90)
        potential_locations = np.argwhere(vessel_mask)
        
        if len(potential_locations) == 0:
            return volume, {'added': False}
        
        # Random location selection
        idx = np.random.randint(len(potential_locations))
        location = potential_locations[idx]
        
        # Random size
        size_range = self.config['augmentation']['synthetic_size_range']
        size = np.random.randint(size_range[0], size_range[1])
        
        # Create spherical aneurysm
        sphere = morphology.ball(size // 2)
        
        # Place aneurysm
        x, y, z = location
        x_start = max(0, x - size // 2)
        x_end = min(volume.shape[0], x + size // 2 + 1)
        y_start = max(0, y - size // 2)
        y_end = min(volume.shape[1], y + size // 2 + 1)
        z_start = max(0, z - size // 2)
        z_end = min(volume.shape[2], z + size // 2 + 1)
        
        # Add with intensity variation
        intensity = np.random.uniform(0.7, 1.0) * volume.max()
        
        # Ensure sphere fits in the region
        sphere_crop = sphere[
            :x_end-x_start,
            :y_end-y_start,
            :z_end-z_start
        ]
        
        augmented[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            augmented[x_start:x_end, y_start:y_end, z_start:z_end],
            sphere_crop * intensity
        )
        
        # Smooth edges
        augmented = gaussian_filter(augmented, sigma=0.5)
        
        info = {
            'added': True,
            'location': location.tolist(),
            'size': size,
            'intensity': intensity
        }
        
        return augmented, info
    
    def process_volume(self, volume: np.ndarray, 
                      augment: bool = False) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline for a single volume.
        
        Args:
            volume: Input 3D medical image
            augment: Whether to apply augmentation
            
        Returns:
            Dictionary with processed volumes
        """
        results = {}
        
        # Step 1: Intensity normalization
        normalized = self.intensity_normalization(volume)
        results['normalized'] = normalized
        
        # Step 2: Vessel enhancement (Frangi filter)
        print("Applying Frangi vesselness filter...")
        vesselness = self.frangi_vesselness_3d(normalized)
        results['vesselness'] = vesselness
        
        # Step 3: Aneurysm-specific enhancement
        print("Enhancing aneurysm regions...")
        aneurysm_enhanced = self.enhance_aneurysm_regions(normalized, vesselness)
        results['aneurysm_enhanced'] = aneurysm_enhanced
        
        # Step 4: Temporal MIP
        mip_enhanced = self.temporal_mip(aneurysm_enhanced)
        results['mip_enhanced'] = mip_enhanced
        
        # Step 5: Optional augmentation
        if augment:
            augmented, aug_info = self.add_synthetic_aneurysm(mip_enhanced, vesselness)
            results['augmented'] = augmented
            results['augmentation_info'] = aug_info
        
        # Step 6: Combined output (weighted combination)
        combined = (
            0.3 * normalized +
            0.3 * vesselness +
            0.4 * aneurysm_enhanced
        )
        results['combined'] = combined
        
        return results
    
    def preprocess_batch(self, volumes: List[np.ndarray], 
                        augment: bool = False) -> List[Dict[str, np.ndarray]]:
        """
        Process multiple volumes in batch.
        
        Args:
            volumes: List of 3D volumes
            augment: Whether to apply augmentation
            
        Returns:
            List of processed volume dictionaries
        """
        results = []
        
        for i, volume in enumerate(volumes):
            print(f"Processing volume {i+1}/{len(volumes)}...")
            processed = self.process_volume(volume, augment=augment)
            results.append(processed)
        
        return results


class MultiTaskPreprocessor:
    """
    Preprocessor optimized for both aneurysm detection and vessel classification.
    """
    
    def __init__(self, aneurysm_pipeline: AneurysmPreprocessingPipeline):
        """
        Initialize multi-task preprocessor.
        
        Args:
            aneurysm_pipeline: Aneurysm preprocessing pipeline
        """
        self.aneurysm_pipeline = aneurysm_pipeline
        
    def create_vessel_specific_features(self, volume: np.ndarray, 
                                       vesselness: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create vessel-specific features for classification.
        
        Args:
            volume: Original volume
            vesselness: Vessel-enhanced volume
            
        Returns:
            Dictionary of vessel-specific features
        """
        features = {}
        
        # ICA features (typically larger vessels)
        ica_scales = [2.0, 2.5, 3.0]
        features['ICA'] = self._extract_vessel_at_scales(volume, ica_scales)
        
        # MCA features (medium vessels)
        mca_scales = [1.5, 2.0, 2.5]
        features['MCA'] = self._extract_vessel_at_scales(volume, mca_scales)
        
        # ACA features (anterior vessels)
        aca_scales = [1.0, 1.5, 2.0]
        features['ACA'] = self._extract_vessel_at_scales(volume, aca_scales)
        
        # Pcom features (smaller communicating vessels)
        pcom_scales = [0.5, 1.0, 1.5]
        features['Pcom'] = self._extract_vessel_at_scales(volume, pcom_scales)
        
        return features
    
    def _extract_vessel_at_scales(self, volume: np.ndarray, 
                                  scales: List[float]) -> np.ndarray:
        """
        Extract vessel features at specific scales.
        
        Args:
            volume: Input volume
            scales: List of scales for vessel extraction
            
        Returns:
            Multi-scale vessel features
        """
        features = []
        
        for scale in scales:
            # Apply Gaussian smoothing at scale
            smoothed = gaussian_filter(volume, sigma=scale)
            
            # Compute gradient magnitude
            gradients = np.gradient(smoothed)
            grad_mag = np.sqrt(sum(g**2 for g in gradients))
            
            features.append(grad_mag)
        
        # Stack features
        return np.stack(features, axis=-1)
    
    def process_for_multitask(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process volume for multi-task learning.
        
        Args:
            volume: Input 3D volume
            
        Returns:
            Dictionary with task-specific features
        """
        # Get aneurysm-focused preprocessing
        aneurysm_results = self.aneurysm_pipeline.process_volume(volume)
        
        # Extract vessel-specific features
        vessel_features = self.create_vessel_specific_features(
            volume, 
            aneurysm_results['vesselness']
        )
        
        # Combine for multi-task output
        output = {
            'aneurysm_features': aneurysm_results['combined'],
            'vessel_features': vessel_features,
            'shared_features': aneurysm_results['vesselness'],
            'raw_normalized': aneurysm_results['normalized']
        }
        
        return output


# Example usage
if __name__ == "__main__":
    # Create synthetic test volume
    print("Creating synthetic test volume...")
    volume = np.random.randn(128, 128, 64) * 100 + 500
    volume[volume < 0] = 0
    
    # Initialize pipeline
    print("\nInitializing preprocessing pipeline...")
    pipeline = AneurysmPreprocessingPipeline()
    
    # Process volume
    print("\nProcessing volume...")
    results = pipeline.process_volume(volume, augment=True)
    
    # Print results
    print("\nProcessing complete! Generated features:")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
        else:
            print(f"  - {key}: {value}")
    
    # Multi-task preprocessing
    print("\n" + "="*60)
    print("Multi-task preprocessing example:")
    print("="*60)
    
    mt_preprocessor = MultiTaskPreprocessor(pipeline)
    mt_results = mt_preprocessor.process_for_multitask(volume)
    
    print("\nMulti-task features generated:")
    for key, value in mt_results.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    - {subkey}: shape {subvalue.shape}")
        elif isinstance(value, np.ndarray):
            print(f"  - {key}: shape {value.shape}")
    
    print("\nPreprocessing pipeline ready for integration!")