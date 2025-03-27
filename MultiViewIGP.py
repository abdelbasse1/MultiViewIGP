import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import numpy as np

class IslamicPatternClusterer:
    """Class for multiview clustering of Islamic geometric patterns with publication-ready outputs."""
    
    def __init__(self, folder_path, output_dir=None):
        """Initialize the clusterer with folder path and output directory."""
        self.folder_path = folder_path
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"islamic_pattern_clustering_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize attributes
        self.images = []
        self.filenames = []
        self.feature_names = ["Color", "Texture", "Deep_VGG16", "Deep_ResNet50", "Deep_EfficientNet", 
                      "Deep_MobileNet", "Deep_DenseNet", "Deep_Combined", "Symmetry"]
        self.features = {name: None for name in self.feature_names}
        self.normalized_features = {name: None for name in self.feature_names}
        self.view_labels = {name: None for name in self.feature_names}
        self.view_scores = {name: {} for name in self.feature_names}
        self.optimal_k = {name: None for name in self.feature_names}
        self.final_labels = None
        self.consensus_matrix = None
        
        # Configure plot styling for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 20
        
        # For reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_images(self):
        """Load all images from the folder path."""
        images = []
        filenames = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        print(f"Loading images from {self.folder_path}...")
        files = [f for f in os.listdir(self.folder_path) 
                if os.path.splitext(f)[1].lower() in valid_extensions]
        
        for filename in files:
            img_path = os.path.join(self.folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    images.append(img)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        self.images = images
        self.filenames = filenames
        print(f"Successfully loaded {len(images)} images.")
        
        # Save list of processed files
        with open(os.path.join(self.output_dir, 'processed_files.txt'), 'w') as f:
            for filename in filenames:
                f.write(f"{filename}\n")
        
        return images, filenames
    
    def extract_color_features(self, image, bins=32):
        """Extract color histogram features."""
        # Resize for consistency
        image = resize(image, (224, 224), preserve_range=True).astype(np.uint8)
        
        # Convert to HSV color space (better for color analysis)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalize and concatenate
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def extract_texture_features(self, image):
        """Extract texture features using HOG and LBP."""
        # Resize and convert to grayscale
        image = resize(image, (224, 224), preserve_range=True).astype(np.uint8)
        gray = rgb2gray(image)
        
        # HOG features
        hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=False, feature_vector=True)
        
        # LBP features
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return np.concatenate([hog_features, hist])
    
    def extract_deep_features(self, batch_size=16):
        """Extract deep features using multiple pre-trained models."""
        print("Initializing deep feature extraction models...")
        
        # Dictionary to hold features from different models
        all_features = {}
        
        # 1. VGG16 features (original)
        print("Extracting VGG16 features...")
        vgg_model = VGG16(weights='DeepModel/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
        vgg_features = self._extract_from_model(vgg_model, batch_size)
        all_features['vgg16'] = vgg_features
        
        # 2. ResNet50 features
        print("Extracting ResNet50 features...")
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        
        resnet_model = ResNet50(weights='DeepModel/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
        resnet_features = self._extract_from_model(resnet_model, batch_size, preprocess_fn=resnet_preprocess)
        all_features['resnet50'] = resnet_features
        
        # 3. EfficientNetB0 features
        print("Extracting EfficientNet features...")
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
        
        efficientnet_model = EfficientNetB0(weights='DeepModel/efficientnetb0_notop.h5', include_top=False, pooling='avg')
        efficientnet_features = self._extract_from_model(efficientnet_model, batch_size, preprocess_fn=efficientnet_preprocess)
        all_features['efficientnet'] = efficientnet_features
        
        # 4. MobileNetV2 features
        print("Extracting MobileNetV2 features...")
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
        
        mobilenet_model = MobileNetV2(weights='DeepModel/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', include_top=False, pooling='avg')
        mobilenet_features = self._extract_from_model(mobilenet_model, batch_size, preprocess_fn=mobilenet_preprocess)
        all_features['mobilenet'] = mobilenet_features
        
        # 5. DenseNet121 features
        print("Extracting DenseNet features...")
        from tensorflow.keras.applications import DenseNet121
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
        
        densenet_model = DenseNet121(weights='DeepModel/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
        densenet_features = self._extract_from_model(densenet_model, batch_size, preprocess_fn=densenet_preprocess)
        all_features['densenet'] = densenet_features
        
        # Concatenate all features
        print("Combining features from all models...")
        combined_features = np.concatenate([
            all_features['vgg16'],
            all_features['resnet50'],
            all_features['efficientnet'],
            all_features['mobilenet'],
            all_features['densenet']
        ], axis=1)
        
        # Save individual model features for potential separate analysis
        for name, features in all_features.items():
            np.save(os.path.join(self.output_dir, f'deep_features_{name}.npy'), features)
        
        # Return the combined features
        return all_features
    
    def _extract_from_model(self, model, batch_size=16, preprocess_fn=preprocess_input):
        """Helper method to extract features from a specific model."""
        features = []
        
        # Process images in batches
        for i in range(0, len(self.images), batch_size):
            batch_images = self.images[i:i+batch_size]
            batch_processed = []
            
            for img in batch_images:
                # Resize to model's input size (224x224 for most models)
                img = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
                # Convert to tensor and preprocess
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_fn(x)
                batch_processed.append(x)
                
            batch_processed = np.vstack(batch_processed)
            batch_features = model.predict(batch_processed, verbose=0)
            features.extend(batch_features)
            
            # Print progress
            print(f"  Processed {min(i+batch_size, len(self.images))}/{len(self.images)} images")
        
        return np.array(features)
    
    def extract_symmetry_features(self, image):
        """Extract symmetry-based features relevant for Islamic patterns."""
        # Resize and convert to grayscale
        image = resize(image, (224, 224), preserve_range=True).astype(np.uint8)
        gray = rgb2gray(image)
        
        # Calculate rotational symmetry features
        features = []
        center = (gray.shape[0] // 2, gray.shape[1] // 2)
        
        # Check similarity at different rotation angles
        angles = [30, 45, 60, 90, 120, 180]  # Common angles in Islamic patterns
        for angle in angles:
            # Rotate image
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, rotation_matrix, gray.shape[::-1])
            
            # Calculate similarity (correlation)
            similarity = np.corrcoef(gray.flatten(), rotated.flatten())[0, 1]
            features.append(similarity)
        
        # Check for horizontal and vertical reflective symmetry
        h_flipped = cv2.flip(gray, 1)  # Horizontal flip
        v_flipped = cv2.flip(gray, 0)  # Vertical flip
        
        h_symmetry = np.corrcoef(gray.flatten(), h_flipped.flatten())[0, 1]
        v_symmetry = np.corrcoef(gray.flatten(), v_flipped.flatten())[0, 1]
        
        features.extend([h_symmetry, v_symmetry])
        
        return np.array(features)
    
    def extract_all_features(self):
        """Extract all feature types for all images."""
        print("Extracting features from all images...")
        
        # 1. Color features
        print("Extracting color features...")
        color_features = np.array([self.extract_color_features(img) for img in self.images])
        self.features["Color"] = color_features
        
        # 2. Texture features
        print("Extracting texture features...")
        texture_features = np.array([self.extract_texture_features(img) for img in self.images])
        self.features["Texture"] = texture_features
        
        # 3. Deep features
        print("Extracting deep features...")
        deep_features_dict = self.extract_deep_features()
        self.features["Deep_VGG16"] = deep_features_dict['vgg16']
        self.features["Deep_ResNet50"] = deep_features_dict['resnet50']
        self.features["Deep_EfficientNet"] = deep_features_dict['efficientnet']
        self.features["Deep_MobileNet"] = deep_features_dict['mobilenet']
        self.features["Deep_DenseNet"] = deep_features_dict['densenet']
        self.features["Deep_Combined"] = np.concatenate(list(deep_features_dict.values()), axis=1)
        
        # 4. Symmetry features
        print("Extracting symmetry features...")
        symmetry_features = np.array([self.extract_symmetry_features(img) for img in self.images])
        self.features["Symmetry"] = symmetry_features
        
        # Save feature dimensions
        feature_dimensions = {
            name: features.shape for name, features in self.features.items()
        }
        
        with open(os.path.join(self.output_dir, 'feature_dimensions.json'), 'w') as f:
            json.dump({k: str(v) for k, v in feature_dimensions.items()}, f, indent=2)
        
        print("Feature extraction complete.")
        print("Feature dimensions:")
        for name, shape in feature_dimensions.items():
            print(f"  {name}: {shape}")
        
        return self.features
    
    def normalize_features(self):
        """Normalize each feature set independently."""
        print("Normalizing features...")
        
        for name, features in self.features.items():
            scaler = StandardScaler()
            self.normalized_features[name] = scaler.fit_transform(features)
        
        return self.normalized_features
    
    def evaluate_clustering(self, features, labels, reference_labels=None):
        """Evaluate clustering with multiple metrics."""
        try:
            sil_score = silhouette_score(features, labels)
        except:
            sil_score = float('nan')
            
        try:
            ch_score = calinski_harabasz_score(features, labels)
        except:
            ch_score = float('nan')
            
        try:
            db_score = davies_bouldin_score(features, labels)
        except:
            db_score = float('nan')
        
        # Calculate ARI if reference labels are provided
        ari_score = float('nan')
        if reference_labels is not None:
            try:
                ari_score = adjusted_rand_score(reference_labels, labels)
            except:
                pass
        
        return {
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score,
            'ari': ari_score
        }
    
    def find_optimal_k(self, n_clusters_range=range(2, 11)):
        """Find optimal number of clusters for each feature type."""
        print("Finding optimal number of clusters for each feature type...")
        
        # Prepare results dictionary
        results = {}
        
        # First pass: compute labels for each feature type and k
        all_labels = {}
        for name, features in self.normalized_features.items():
            all_labels[name] = {}
            for k in n_clusters_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                all_labels[name][k] = {'kmeans': kmeans.fit_predict(features)}
                
                try:
                    spectral = SpectralClustering(n_clusters=k, random_state=42, 
                                               affinity='nearest_neighbors', n_neighbors=min(10, len(features)-1))
                    all_labels[name][k]['spectral'] = spectral.fit_predict(features)
                except Exception as e:
                    print(f"    Error with spectral clustering for k={k}: {e}")
        
        # Second pass: evaluate with metrics including ARI
        for name, features in self.normalized_features.items():
            print(f"Analyzing {name} features...")
            
            # Metrics per k
            k_metrics = {k: {'kmeans': {}, 'spectral': {}} for k in n_clusters_range}
            
            for k in n_clusters_range:
                print(f"  Testing with k={k}...")
                
                # Evaluate KMeans
                kmeans_labels = all_labels[name][k]['kmeans']
                kmeans_metrics = self.evaluate_clustering(features, kmeans_labels)
                
                # Calculate ARI against other views
                ari_scores = []
                for other_name in self.normalized_features.keys():
                    if other_name != name:
                        for other_k in n_clusters_range:
                            for method in ['kmeans', 'spectral']:
                                if method in all_labels[other_name][other_k]:
                                    other_labels = all_labels[other_name][other_k][method]
                                    ari = adjusted_rand_score(kmeans_labels, other_labels)
                                    ari_scores.append(ari)
                
                if ari_scores:
                    kmeans_metrics['ari_mean'] = np.mean(ari_scores)
                    kmeans_metrics['ari_max'] = np.max(ari_scores)
                else:
                    kmeans_metrics['ari_mean'] = float('nan')
                    kmeans_metrics['ari_max'] = float('nan')
                    
                k_metrics[k]['kmeans'] = kmeans_metrics
                
                # Evaluate Spectral clustering
                if 'spectral' in all_labels[name][k]:
                    spectral_labels = all_labels[name][k]['spectral']
                    spectral_metrics = self.evaluate_clustering(features, spectral_labels)
                    
                    # Calculate ARI against other views
                    ari_scores = []
                    for other_name in self.normalized_features.keys():
                        if other_name != name:
                            for other_k in n_clusters_range:
                                for method in ['kmeans', 'spectral']:
                                    if method in all_labels[other_name][other_k]:
                                        other_labels = all_labels[other_name][other_k][method]
                                        ari = adjusted_rand_score(spectral_labels, other_labels)
                                        ari_scores.append(ari)
                    
                    if ari_scores:
                        spectral_metrics['ari_mean'] = np.mean(ari_scores)
                        spectral_metrics['ari_max'] = np.max(ari_scores)
                    else:
                        spectral_metrics['ari_mean'] = float('nan')
                        spectral_metrics['ari_max'] = float('nan')
                        
                    k_metrics[k]['spectral'] = spectral_metrics
                else:
                    k_metrics[k]['spectral'] = {'silhouette': float('nan'), 
                                              'calinski_harabasz': float('nan'), 
                                              'davies_bouldin': float('nan'),
                                              'ari_mean': float('nan'),
                                              'ari_max': float('nan')}
            
            # Find best k based on silhouette score
            best_sil = -1
            best_k = 2
            best_method = 'kmeans'
            best_labels = None
            
            for k in n_clusters_range:
                # Check KMeans
                if k_metrics[k]['kmeans']['silhouette'] > best_sil:
                    best_sil = k_metrics[k]['kmeans']['silhouette']
                    best_k = k
                    best_method = 'kmeans'
                
                # Check Spectral
                if not np.isnan(k_metrics[k]['spectral']['silhouette']) and k_metrics[k]['spectral']['silhouette'] > best_sil:
                    best_sil = k_metrics[k]['spectral']['silhouette']
                    best_k = k
                    best_method = 'spectral'
            
            # Store optimal k
            self.optimal_k[name] = best_k
            
            # Compute final labels with optimal k and method
            if best_method == 'kmeans':
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
            else:
                spectral = SpectralClustering(n_clusters=best_k, random_state=42, 
                                            affinity='nearest_neighbors', n_neighbors=min(10, len(features)-1))
                labels = spectral.fit_predict(features)
            
            # Store labels and metrics
            self.view_labels[name] = labels
            self.view_scores[name] = k_metrics
            
            print(f"  Optimal k for {name}: {best_k} using {best_method} clustering")
            print(f"  Silhouette score: {best_sil:.4f}")
            
            # Plot silhouette scores
            self.plot_metrics(name, k_metrics)
            
            # Store results
            results[name] = {
                'optimal_k': best_k,
                'method': best_method,
                'metrics': k_metrics[best_k][best_method]
            }
        
        # Save results
        with open(os.path.join(self.output_dir, 'optimal_k_results.json'), 'w') as f:
            # Convert ndarray to list for JSON serialization
            serializable_results = {
                k: {
                    'optimal_k': v['optimal_k'],
                    'method': v['method'],
                    'metrics': {
                        mk: float(mv) for mk, mv in v['metrics'].items()
                    }
                } for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=2)
        
        return results
    
    import os
    import matplotlib.pyplot as plt
    
    def plot_metrics(self, feature_name, k_metrics):
        """Plot clustering metrics for different k values."""
        plt.figure(figsize=(12, 8))  # Adjust figure size
    
        # Get k values
        k_values = sorted(k_metrics.keys())
        
        # Prepare data
        kmeans_sil = [k_metrics[k]['kmeans']['silhouette'] for k in k_values]
        spectral_sil = [k_metrics[k]['spectral']['silhouette'] for k in k_values]
    
        kmeans_ch = [k_metrics[k]['kmeans']['calinski_harabasz'] for k in k_values]
        spectral_ch = [k_metrics[k]['spectral']['calinski_harabasz'] for k in k_values]
    
        kmeans_db = [k_metrics[k]['kmeans']['davies_bouldin'] for k in k_values]
        spectral_db = [k_metrics[k]['spectral']['davies_bouldin'] for k in k_values]
    
        kmeans_ari = [k_metrics[k]['kmeans'].get('ari_mean', float('nan')) for k in k_values]
        spectral_ari = [k_metrics[k]['spectral'].get('ari_mean', float('nan')) for k in k_values]
    
        # Create a 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
        # Silhouette Score
        axes[0, 0].plot(k_values, kmeans_sil, 'o-', label='KMeans')
        axes[0, 0].plot(k_values, spectral_sil, 's-', label='Spectral')
        axes[0, 0].set_xlabel('Number of clusters (k)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Scores')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
    
        # Calinski-Harabasz Index
        axes[0, 1].plot(k_values, kmeans_ch, 'o-', label='KMeans')
        axes[0, 1].plot(k_values, spectral_ch, 's-', label='Spectral')
        axes[0, 1].set_xlabel('Number of clusters (k)')
        axes[0, 1].set_ylabel('Calinski-Harabasz Index')
        axes[0, 1].set_title('Calinski-Harabasz Index')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
    
        # Davies-Bouldin Index
        axes[1, 0].plot(k_values, kmeans_db, 'o-', label='KMeans')
        axes[1, 0].plot(k_values, spectral_db, 's-', label='Spectral')
        axes[1, 0].set_xlabel('Number of clusters (k)')
        axes[1, 0].set_ylabel('Davies-Bouldin Index')
        axes[1, 0].set_title('Davies-Bouldin Index')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
        # Adjusted Rand Index (ARI)
        axes[1, 1].plot(k_values, kmeans_ari, 'o-', label='KMeans')
        axes[1, 1].plot(k_values, spectral_ari, 's-', label='Spectral')
        axes[1, 1].set_xlabel('Number of clusters (k)')
        axes[1, 1].set_ylabel('Average ARI with Other Views')
        axes[1, 1].set_title('Adjusted Rand Index')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'metrics_{feature_name}.png'), dpi=300)
        plt.close()
    
    def create_consensus_matrix(self):
        """Create consensus matrix from all feature views."""
        print("Creating consensus matrix...")
        
        n_samples = len(self.images)
        consensus_matrix = np.zeros((n_samples, n_samples))
        
        # Build consensus matrix based on co-occurrence
        for name, labels in self.view_labels.items():
            print(f"  Adding {name} view to consensus...")
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
        
        # Normalize consensus matrix
        consensus_matrix /= len(self.feature_names)
        self.consensus_matrix = consensus_matrix
        
        # Save consensus matrix
        np.save(os.path.join(self.output_dir, 'consensus_matrix.npy'), consensus_matrix)
        
        # Visualize consensus matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(consensus_matrix, cmap='viridis')
        plt.title("Consensus Matrix")
        plt.savefig(os.path.join(self.output_dir, 'consensus_matrix.png'), dpi=300)
        plt.close()
        
        return consensus_matrix
    
    def perform_final_clustering(self):
        """Perform final clustering on consensus matrix."""
        print("Performing final clustering...")
        
        # Compute distance matrix
        distance_matrix = 1 - self.consensus_matrix
        
        # Try different numbers of clusters for spectral clustering on consensus matrix
        n_clusters_range = range(5, 11)
        best_sil = -1
        best_k = 2
        best_labels = None
        ari_with_views = {}
        
        for k in n_clusters_range:
            print(f"  Testing with k={k}...")
            
            # Spectral clustering on consensus matrix
            spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='precomputed')
            try:
                labels = spectral.fit_predict(self.consensus_matrix)
                sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
                
                # Calculate ARI with individual view clusterings
                view_ari = {}
                for name, view_labels in self.view_labels.items():
                    ari = adjusted_rand_score(view_labels, labels)
                    view_ari[name] = ari
                    
                avg_ari = np.mean(list(view_ari.values()))
                
                print(f"    Silhouette score: {sil_score:.4f}, Avg ARI with views: {avg_ari:.4f}")
                
                if sil_score > best_sil:
                    best_sil = sil_score
                    best_k = k
                    best_labels = labels
                    ari_with_views = view_ari
            except Exception as e:
                print(f"    Error with k={k}: {e}")
        
        print(f"Optimal number of clusters: {best_k}")
        print(f"Silhouette score: {best_sil:.4f}")
        print("ARI with individual views:")
        for name, ari in ari_with_views.items():
            print(f"  - {name}: {ari:.4f}")
        
        self.final_labels = best_labels
        self.ari_with_views = ari_with_views
        
        # Save final labels
        np.save(os.path.join(self.output_dir, 'final_labels.npy'), best_labels)
        
        # Save ARI values
        with open(os.path.join(self.output_dir, 'ari_with_views.json'), 'w') as f:
            json.dump({k: float(v) for k, v in ari_with_views.items()}, f, indent=2)
        
        # Create cluster assignments file
        cluster_assignments = pd.DataFrame({
            'filename': self.filenames,
            'cluster': best_labels
        })
        cluster_assignments.to_csv(os.path.join(self.output_dir, 'cluster_assignments.csv'), index=False)
        
        return best_labels
    
    def visualize_clusters(self, max_per_cluster=5):
        """Visualize clustering results with publication quality figures."""
        print("Visualizing clusters...")
        
        unique_labels = np.unique(self.final_labels)
        n_clusters = len(unique_labels)
        
        # Create cluster directory
        clusters_dir = os.path.join(self.output_dir, 'clusters')
        os.makedirs(clusters_dir, exist_ok=True)
        
        # Create overview figure
        fig = plt.figure(figsize=(15, n_clusters * 3))
        
        for i, label in enumerate(unique_labels):
            indices = np.where(self.final_labels == label)[0]
            n_samples = min(max_per_cluster, len(indices))
            
            # Save cluster members
            cluster_dir = os.path.join(clusters_dir, f'cluster_{label}')
            os.makedirs(cluster_dir, exist_ok=True)
            
            with open(os.path.join(cluster_dir, 'members.txt'), 'w') as f:
                for idx in indices:
                    f.write(f"{self.filenames[idx]}\n")
            
            # Create individual cluster visualization
            plt.figure(figsize=(15, 3))
            for j in range(n_samples):
                plt.subplot(1, n_samples, j + 1)
                plt.imshow(self.images[indices[j]])
                plt.title(f"{self.filenames[indices[j]]}")
                plt.axis('off')
            
            plt.suptitle(f"Cluster {label} ({len(indices)} images)")
            plt.tight_layout()
            plt.savefig(os.path.join(cluster_dir, 'preview.png'), dpi=300)
            plt.close()
            
            # Add to overview figure
            for j in range(n_samples):
                ax = fig.add_subplot(n_clusters, max_per_cluster, i * max_per_cluster + j + 1)
                ax.imshow(self.images[indices[j]])
                ax.set_title(f"C{label}: {self.filenames[indices[j]]}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_overview.png'), dpi=300)
        plt.close()
        
        print(f"Visualization saved to {self.output_dir}")
    
    def visualize_feature_spaces(self):
        """Visualize feature spaces with dimension reduction."""
        print("Visualizing feature spaces...")
        
        for name, features in self.normalized_features.items():
            print(f"  Processing {name} features...")
            
            # PCA for initial dimension reduction
            n_components = min(50, features.shape[1])
            pca = PCA(n_components=n_components)
            
            try:
                # For very high-dimensional features, first reduce with PCA
                if features.shape[1] > 50:
                    reduced_features = pca.fit_transform(features)
                else:
                    reduced_features = features
                
                # t-SNE for visualization
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
                tsne_results = tsne.fit_transform(reduced_features)
                
                # Plot t-SNE results colored by final clusters
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                                    c=self.final_labels, cmap='viridis', 
                                    alpha=0.8, s=100)
                plt.colorbar(scatter, label='Cluster')
                plt.title(f't-SNE Visualization of {name} Features')
                plt.savefig(os.path.join(self.output_dir, f'tsne_{name}.png'), dpi=300)
                plt.close()
                
                # Save t-SNE results
                np.save(os.path.join(self.output_dir, f'tsne_{name}.npy'), tsne_results)
            
            except Exception as e:
                print(f"    Error visualizing {name} features: {e}")
    
    def create_cluster_profile(self):
        """Create profile of each cluster with key characteristics."""
        print("Creating cluster profiles...")
        
        unique_labels = np.unique(self.final_labels)
        
        # Prepare cluster profiles
        profiles = {}
        
        for label in unique_labels:
            indices = np.where(self.final_labels == label)[0]
            
            # Basic statistics
            profile = {
                'count': len(indices),
                'percentage': len(indices) / len(self.images) * 100,
                'members': [self.filenames[i] for i in indices]
            }
            
            # Feature analysis for each cluster
            for name, features in self.normalized_features.items():
                # Extract feature vectors for this cluster
                cluster_features = features[indices]
                
                # Calculate mean feature vector
                mean_vector = np.mean(cluster_features, axis=0)
                
                # Calculate standard deviation
                std_vector = np.std(cluster_features, axis=0)
                
                # For symmetry features, provide interpretable values
                if name == "Symmetry":
                    symmetry_angles = [30, 45, 60, 90, 120, 180, "h_flip", "v_flip"]
                    symmetry_profile = {}
                    
                    for i, angle in enumerate(symmetry_angles):
                        if i < len(mean_vector):
                            sym_value = mean_vector[i]
                            symmetry_profile[f"{angle}"] = float(sym_value)
                    
                    profile[f"{name}_symmetry"] = symmetry_profile
                
                # Store mean and std dev for feature type
                profile[f"{name}_mean"] = float(np.mean(np.abs(mean_vector)))
                profile[f"{name}_std"] = float(np.mean(std_vector))
            
            profiles[int(label)] = profile
        
        # Save profiles
        with open(os.path.join(self.output_dir, 'cluster_profiles.json'), 'w') as f:
            # Convert to serializable format
            serializable_profiles = {}
            for k, v in profiles.items():
                serializable_profiles[str(k)] = {
                    kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) 
                    for kk, vv in v.items()
                }
            json.dump(serializable_profiles, f, indent=2)
        
        # Create summary table
        summary_data = []
        for label in unique_labels:
            profile = profiles[int(label)]
            row = {
                'Cluster': label,
                'Count': profile['count'],
                'Percentage': f"{profile['percentage']:.1f}%"
            }
            
            # Add feature means
            for name in self.feature_names:
                row[f"{name} Mean"] = f"{profile[f'{name}_mean']:.4f}"
            
            # Add symmetry features if available
            if 'Symmetry_symmetry' in profile:
                for angle, value in profile['Symmetry_symmetry'].items():
                    row[f"Sym {angle}"] = f"{value:.2f}"
            
            summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, 'cluster_summary.csv'), index=False)
        
        return profiles
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for paper inclusion."""
        print("Generating LaTeX tables...")
        
        # 1. Cluster summary table
        cluster_summary = pd.read_csv(os.path.join(self.output_dir, 'cluster_summary.csv'))
        
        with open(os.path.join(self.output_dir, 'cluster_summary.tex'), 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Summary of Islamic Pattern Clusters}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\n")
            f.write("Cluster & Count & Percentage & Key Symmetry Features \\\\\n")
            f.write("\\hline\n")
            
            unique_labels = np.unique(self.final_labels)
            
            for label in unique_labels:
                cluster_data = cluster_summary[cluster_summary['Cluster'] == label]
                
                # Extract symmetry features if available
                sym_features = []
                for col in cluster_data.columns:
                    if col.startswith('Sym '):
                        #value = float(cluster_data[col].values[0].replace(',', '.').strip())
                        value_raw = cluster_data[col].values[0]
                        if isinstance(value_raw, str):
                            value = float(value_raw.replace(',', '.').strip())
                        else:
                            value = float(value_raw)
                        if value > 0.7:  # Only report strong symmetry
                            angle = col.replace('Sym ', '')
                            sym_features.append(f"{angle}: {value:.2f}")
                
                sym_text = ", ".join(sym_features) if sym_features else "None dominant"
                
                f.write(f"{int(label)} & {int(cluster_data['Count'])} & {cluster_data['Percentage'].values[0]} & {sym_text} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:cluster_summary}\n")
            f.write("\\end{table}\n")
        
        # 2. Feature evaluation table
        with open(os.path.join(self.output_dir, 'feature_evaluation.tex'), 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Evaluation of Different Feature Types for Islamic Pattern Clustering}\n")
            f.write("\\begin{tabular}{lccccc}\n")  # Added one more column for ARI
            f.write("\\hline\n")
            f.write("Feature Type & Optimal k & Silhouette Score & Calinski-Harabasz & Davies-Bouldin & ARI with Final \\\\\n")
            f.write("\\hline\n")
            
            # Load optimal k results
            with open(os.path.join(self.output_dir, 'optimal_k_results.json'), 'r') as fr:
                results = json.load(fr)
            
            # Load ARI values 
            with open(os.path.join(self.output_dir, 'ari_with_views.json'), 'r') as fr:
                ari_values = json.load(fr)
            
            for name, result in results.items():
                metrics = result['metrics']
                ari_value = ari_values.get(name, 0.0)
                f.write(f"{name} & {result['optimal_k']} & {metrics['silhouette']:.4f} & ")
                f.write(f"{metrics['calinski_harabasz']:.1f} & {metrics['davies_bouldin']:.4f} & {ari_value:.4f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:feature_evaluation}\n")
            f.write("\\end{table}\n")
        
        print("LaTeX tables generated.")
    
    def generate_report(self):
        """Generate comprehensive PDF report with results."""
        print("Generating comprehensive report...")
        
        # Collect cluster data
        cluster_data = []
        unique_labels = np.unique(self.final_labels)
        
        for label in unique_labels:
            indices = np.where(self.final_labels == label)[0]
            cluster_data.append({
                'label': label,
                'count': len(indices),
                'percentage': len(indices) / len(self.images) * 100,
                'samples': [self.filenames[i] for i in indices[:5]]
            })
        
        # Create markdown report
        report_path = os.path.join(self.output_dir, 'clustering_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Multiview Clustering Analysis of Islamic Geometric Patterns\n\n")
            
            f.write("## 1. Overview\n\n")
            f.write(f"- **Dataset**: {len(self.images)} images from {self.folder_path}\n")
            f.write(f"- **Clustering Method**: Multiview consensus clustering\n")
            f.write(f"- **Feature Types**: {', '.join(self.feature_names)}\n")
            f.write(f"- **Number of Clusters**: {len(unique_labels)}\n\n")
            
            f.write("## 2. Feature Analysis\n\n")
            
            for name in self.feature_names:
                f.write(f"### {name} Features\n\n")
                f.write(f"- **Dimension**: {self.features[name].shape[1]}\n")
                f.write(f"- **Optimal k**: {self.optimal_k[name]}\n")
                f.write(f"- **Silhouette Score**: {self.view_scores[name][self.optimal_k[name]]['kmeans']['silhouette']:.4f}\n\n")
                f.write(f"![{name} Metrics](metrics_{name}.png)\n\n")
                f.write(f"![{name} t-SNE](tsne_{name}.png)\n\n")
            
            f.write("## 3. Consensus Matrix\n\n")
            f.write("The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.\n\n")
            f.write("![Consensus Matrix](consensus_matrix.png)\n\n")
            
            # Add this section to the report generation
            f.write("## 3.5. Agreement Between Views (ARI)\n\n")
            f.write("The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.\n\n")
            
            # Create and save ARI comparison plot
            if hasattr(self, 'ari_with_views'):
                plt.figure(figsize=(10, 6))
                views = list(self.ari_with_views.keys())
                ari_values = [self.ari_with_views[v] for v in views]
                
                bars = plt.bar(views, ari_values)
                plt.xlabel('Feature View')
                plt.ylabel('ARI with Final Clustering')
                plt.xticks(rotation=60)

                plt.title('Agreement Between Feature Views and Consensus Clustering')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom')
                
                plt.ylim(0, 1.0)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'ari_comparison.png'), dpi=300)
                plt.close()
                
                f.write("![ARI Comparison](ari_comparison.png)\n\n")
                
                # Add interpretation
                max_view = max(self.ari_with_views.items(), key=lambda x: x[1])
                min_view = min(self.ari_with_views.items(), key=lambda x: x[1])
                
                f.write(f"The {max_view[0]} features show the highest agreement with the final consensus clustering (ARI = {max_view[1]:.2f}), ")
                f.write(f"while {min_view[0]} features show the lowest agreement (ARI = {min_view[1]:.2f}).\n\n")
            f.write("## 4. Clustering Results\n\n")
            f.write("![Cluster Overview](cluster_overview.png)\n\n")
            
            f.write("### Cluster Summary\n\n")
            f.write("| Cluster | Count | % of Dataset | Key Characteristics |\n")
            f.write("|---------|-------|-------------|---------------------|\n")
            
            # Load cluster profiles
            with open(os.path.join(self.output_dir, 'cluster_profiles.json'), 'r') as fr:
                profiles = json.load(fr)
            
            for cluster in sorted([int(c) for c in profiles.keys()]):
                profile = profiles[str(cluster)]
                
                # Extract key characteristics
                key_chars = []
                
                # Check for strong symmetry
                if 'Symmetry_symmetry' in profile:
                    for angle, value in profile['Symmetry_symmetry'].items():
                        if value > 0.7:
                            key_chars.append(f"{angle} symmetry: {value:.2f}")
                
                # Build table row
                chars_text = ", ".join(key_chars[:3]) if key_chars else "No dominant characteristics"
                
                f.write(f"| {cluster} | {profile['count']} | {profile['percentage']:.1f}% | {chars_text} |\n")
            
            f.write("\n\n")
            
            f.write("## 5. Detailed Cluster Analysis\n\n")
            
            for label in unique_labels:
                f.write(f"### Cluster {label}\n\n")
                f.write(f"![Cluster {label}](clusters/cluster_{label}/preview.png)\n\n")
                
                indices = np.where(self.final_labels == label)[0]
                f.write(f"- **Count**: {len(indices)} images ({len(indices)/len(self.images)*100:.1f}% of dataset)\n")
                
                # Extract representative patterns
                f.write(f"- **Representative patterns**: {', '.join([self.filenames[i] for i in indices[:5]])}\n")
                
                # Symmetry analysis
                if f'Symmetry_symmetry' in profiles[str(label)]:
                    f.write("- **Symmetry analysis**:\n")
                    for angle, value in profiles[str(label)]['Symmetry_symmetry'].items():
                        f.write(f"  - {angle}: {value:.2f}\n")
                
                f.write("\n")
            
            f.write("## 6. Conclusion\n\n")
            f.write("This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. ")
            f.write("By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters ")
            f.write("with specific geometric and visual characteristics.\n\n")
            
            f.write("These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative ")
            f.write("classification of pattern styles based on their mathematical and visual properties.\n\n")
        
        print(f"Report generated at {report_path}")
        
        return report_path
    
    def run_pipeline(self, n_clusters_range=range(2, 11)):
        """Run the complete clustering pipeline."""
        # 1. Load images
        self.load_images()
        
        if len(self.images) == 0:
            print("No images found. Exiting.")
            return
        
        # 2. Extract features
        self.extract_all_features()
        
        # 3. Normalize features
        self.normalize_features()
        
        # 4. Find optimal k for each feature type
        self.find_optimal_k(n_clusters_range)
        
        # 5. Create consensus matrix
        self.create_consensus_matrix()
        
        # 6. Perform final clustering
        self.perform_final_clustering()
        
        # 7. Visualize clusters
        self.visualize_clusters()
        
        # 8. Visualize feature spaces
        self.visualize_feature_spaces()
        
        # 9. Create cluster profiles
        self.create_cluster_profile()
        
        # 10. Generate LaTeX tables
        self.generate_latex_tables()
        
        # 11. Generate report
        report_path = self.generate_report()
        
        print("\nClustering pipeline completed successfully!")
        print(f"All results saved to: {self.output_dir}")
        print(f"Summary report: {report_path}")
        
        return {
            'output_dir': self.output_dir,
            'n_clusters': len(np.unique(self.final_labels)),
            'n_images': len(self.images),
            'report_path': report_path
        }
if __name__ == "__main__":
    
    def cross_region_analysis(regions):
        # Store results from each region
        all_results = {}
        
        for region in regions:
            print(f"\n\nProcessing region: {region}")
            clusterer = IslamicPatternClusterer(f"data/{region}", f"output_{region}")
            results = clusterer.run_pipeline()
            all_results[region] = results
        
        # Create comparative analysis
        output_dir = "cross_region_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Region': [],
            'Number of Images': [],
            'Number of Clusters': [],
            'Top Feature': [],
            'Dominant Symmetry': []
        })
        
        # Populate with data from each region
        for region, results in all_results.items():
            # Load the cluster profiles
            with open(os.path.join(f"output_{region}", 'cluster_profiles.json'), 'r') as f:
                profiles = json.load(f)
            
            # Find dominant symmetry across all clusters
            all_symmetry = {}
            for cluster_id, profile in profiles.items():
                if 'Symmetry_symmetry' in profile:
                    for angle, value in profile['Symmetry_symmetry'].items():
                        if angle not in all_symmetry:
                            all_symmetry[angle] = []
                        all_symmetry[angle].append(value)
            
            # Average symmetry values
            avg_symmetry = {a: np.mean(v) for a, v in all_symmetry.items()}
            dominant_sym = max(avg_symmetry.items(), key=lambda x: x[1]) if avg_symmetry else ('None', 0)
            
            new_row = pd.DataFrame([{
                'Region': region,
                'Number of Images': results['n_images'],
                'Number of Clusters': results['n_clusters'],
                'Top Feature': 'symmetry',  # You'd need to determine this from results
                'Dominant Symmetry': f"{dominant_sym[0]} ({dominant_sym[1]:.2f})"
            }])
            
            comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)

        
        # Save comparison
        comparison_df.to_csv(os.path.join(output_dir, 'region_comparison.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Region', y='Number of Clusters', data=comparison_df)
        plt.title('Number of Pattern Clusters by Region')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clusters_by_region.png'), dpi=300)
        
        return comparison_df
    
    # Run cross-region analysis
    # data/D1
    D1regions = ['egypt', 'france', 'india', 'iran', 'morocco', 'spain', 'syria', 'transoxiana', 'turkey']
    # Others/D2
    D2regions = ['egypt', 'india', 'iran', 'morocco', 'spain', 'syria', 'turkey','asia']
    # data_Conf/D3
    regions_gc = ['gc_india', 'gc_iran', 'gc_morocco', 'gc_transoxiana']
    regions_ii = ['ii_portugal', 'ii_qatar', 'ii_rajastan', 'ii_sicily']
    
    cross_region_analysis(D1regions)
    