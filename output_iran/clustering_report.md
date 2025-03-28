# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 639 images from data/iran
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 5

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 2
- **Silhouette Score**: 0.1295

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 2
- **Silhouette Score**: 0.2170

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.0951

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 2
- **Silhouette Score**: 0.0980

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0468

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0933

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 2
- **Silhouette Score**: 0.0492

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 2
- **Silhouette Score**: 0.0404

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 2
- **Silhouette Score**: 0.3029

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_ResNet50 features show the highest agreement with the final consensus clustering (ARI = 0.32), while Texture features show the lowest agreement (ARI = 0.00).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 87 | 13.6% | No dominant characteristics |
| 1 | 90 | 14.1% | 30 symmetry: 1.23, 45 symmetry: 1.16, 60 symmetry: 1.03 |
| 2 | 147 | 23.0% | No dominant characteristics |
| 3 | 109 | 17.1% | No dominant characteristics |
| 4 | 206 | 32.2% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 87 images (13.6% of dataset)
- **Representative patterns**: ira_0836.jpg, ira_0407.jpg, ira_1335.jpg, ira_2403.jpg, ira_0412.jpg
- **Symmetry analysis**:
  - 30: 0.38
  - 45: 0.22
  - 60: 0.07
  - 90: -0.17
  - 120: -0.39
  - 180: -0.35
  - h_flip: -0.09
  - v_flip: 0.03

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 90 images (14.1% of dataset)
- **Representative patterns**: ira_1527.jpg, ira_1334.jpg, ira_1526.jpg, ira_0604.jpg, ira_1531.jpg
- **Symmetry analysis**:
  - 30: 1.23
  - 45: 1.16
  - 60: 1.03
  - 90: 0.21
  - 120: -0.20
  - 180: -0.84
  - h_flip: 0.57
  - v_flip: -0.57

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 147 images (23.0% of dataset)
- **Representative patterns**: ira_2210.jpg, ira_0215.jpg, ira_0229.jpg, ira_1123.jpg, ira_2614.jpg
- **Symmetry analysis**:
  - 30: -0.09
  - 45: 0.01
  - 60: 0.11
  - 90: 0.07
  - 120: 0.34
  - 180: 0.07
  - h_flip: -0.13
  - v_flip: 0.01

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 109 images (17.1% of dataset)
- **Representative patterns**: ira_0605.jpg, ira_2833.jpg, ira_2601.jpg, ira_1917.jpg, ira_0837.jpg
- **Symmetry analysis**:
  - 30: -0.55
  - 45: -0.65
  - 60: -0.68
  - 90: -0.17
  - 120: -0.22
  - 180: 0.44
  - h_flip: 0.18
  - v_flip: 0.16

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 206 images (32.2% of dataset)
- **Representative patterns**: ira_0611.jpg, ira_1533.jpg, ira_1902.jpg, ira_1916.jpg, ira_0201.jpg
- **Symmetry analysis**:
  - 30: -0.35
  - 45: -0.26
  - 60: -0.20
  - 90: 0.01
  - 120: 0.13
  - 180: 0.24
  - h_flip: -0.21
  - v_flip: 0.14

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

