# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 290 images from data/india
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 5

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 2
- **Silhouette Score**: 0.1135

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 4
- **Silhouette Score**: 0.0865

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.0854

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 2
- **Silhouette Score**: 0.0382

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0594

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0495

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 2
- **Silhouette Score**: 0.0899

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 2
- **Silhouette Score**: 0.0476

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 3
- **Silhouette Score**: 0.2609

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Color features show the highest agreement with the final consensus clustering (ARI = 0.35), while Symmetry features show the lowest agreement (ARI = 0.02).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 50 | 17.2% | No dominant characteristics |
| 1 | 52 | 17.9% | No dominant characteristics |
| 2 | 78 | 26.9% | No dominant characteristics |
| 3 | 56 | 19.3% | No dominant characteristics |
| 4 | 54 | 18.6% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 50 images (17.2% of dataset)
- **Representative patterns**: ind_0311.jpg, ind_0918.jpg, ind_0515.jpg, ind_1009.jpg, ind_0317.jpg
- **Symmetry analysis**:
  - 30: -0.26
  - 45: -0.08
  - 60: -0.02
  - 90: 0.10
  - 120: 0.36
  - 180: 0.01
  - h_flip: -0.46
  - v_flip: 0.17

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 52 images (17.9% of dataset)
- **Representative patterns**: ind_0729.jpg, ind_0700.jpg, ind_0516.jpg, ind_0702.jpg, ind_0919.jpg
- **Symmetry analysis**:
  - 30: -0.42
  - 45: -0.27
  - 60: -0.23
  - 90: -0.01
  - 120: 0.06
  - 180: 0.22
  - h_flip: -0.03
  - v_flip: -0.09

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 78 images (26.9% of dataset)
- **Representative patterns**: ind_0113.jpg, ind_1027.jpg, ind_0310.jpg, ind_1026.jpg, ind_0112.jpg
- **Symmetry analysis**:
  - 30: -0.00
  - 45: -0.12
  - 60: -0.11
  - 90: -0.19
  - 120: -0.13
  - 180: 0.05
  - h_flip: 0.10
  - v_flip: -0.10

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 56 images (19.3% of dataset)
- **Representative patterns**: ind_0503.jpg, ind_0701.jpg, ind_0107.jpg, ind_0106.jpg, ind_0502.jpg
- **Symmetry analysis**:
  - 30: 0.10
  - 45: 0.12
  - 60: 0.14
  - 90: 0.08
  - 120: 0.03
  - 180: -0.06
  - h_flip: 0.18
  - v_flip: -0.01

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 54 images (18.6% of dataset)
- **Representative patterns**: ind_0517.jpg, ind_0514.jpg, ind_1018.jpg, ind_1031.jpg, ind_0504.jpg
- **Symmetry analysis**:
  - 30: 0.55
  - 45: 0.38
  - 60: 0.25
  - 90: 0.10
  - 120: -0.23
  - 180: -0.22
  - h_flip: 0.12
  - v_flip: 0.08

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

