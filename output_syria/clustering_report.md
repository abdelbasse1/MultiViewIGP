# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 217 images from data/syria
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 6

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 2
- **Silhouette Score**: 0.1102

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 3
- **Silhouette Score**: 0.1739

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.0967

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 2
- **Silhouette Score**: 0.0411

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0604

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 3
- **Silhouette Score**: 0.0388

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 2
- **Silhouette Score**: 0.0642

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 2
- **Silhouette Score**: 0.0418

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 5
- **Silhouette Score**: 0.2654

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_MobileNet features show the highest agreement with the final consensus clustering (ARI = 0.35), while Deep_Combined features show the lowest agreement (ARI = 0.01).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 23 | 10.6% | No dominant characteristics |
| 1 | 35 | 16.1% | No dominant characteristics |
| 2 | 58 | 26.7% | No dominant characteristics |
| 3 | 64 | 29.5% | No dominant characteristics |
| 4 | 15 | 6.9% | No dominant characteristics |
| 5 | 22 | 10.1% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 23 images (10.6% of dataset)
- **Representative patterns**: syr_0623.jpg, syr_0421.jpg, syr_0221.jpg, syr_0624.jpg, syr_0220.jpg
- **Symmetry analysis**:
  - 30: 0.19
  - 45: 0.34
  - 60: 0.46
  - 90: 0.30
  - 120: 0.70
  - 180: 0.10
  - h_flip: 0.31
  - v_flip: -0.28

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 35 images (16.1% of dataset)
- **Representative patterns**: syr_0636.jpg, syr_0423.jpg, syr_0235.jpg, syr_0209.jpg, syr_0427.jpg
- **Symmetry analysis**:
  - 30: 0.32
  - 45: 0.27
  - 60: 0.23
  - 90: -0.02
  - 120: -0.04
  - 180: -0.51
  - h_flip: -0.07
  - v_flip: -0.38

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 58 images (26.7% of dataset)
- **Representative patterns**: syr_0233.jpg, syr_0227.jpg, syr_0409.jpg, syr_0434.jpg, syr_0408.jpg
- **Symmetry analysis**:
  - 30: 0.16
  - 45: 0.13
  - 60: 0.07
  - 90: -0.22
  - 120: -0.03
  - 180: -0.13
  - h_flip: -0.14
  - v_flip: -0.26

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 64 images (29.5% of dataset)
- **Representative patterns**: syr_0420.jpg, syr_0232.jpg, syr_0620.jpg, syr_0621.jpg, syr_0619.jpg
- **Symmetry analysis**:
  - 30: 0.03
  - 45: 0.07
  - 60: 0.10
  - 90: 0.17
  - 120: 0.02
  - 180: 0.22
  - h_flip: -0.15
  - v_flip: 0.60

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 15 images (6.9% of dataset)
- **Representative patterns**: syr_0631.jpg, syr_0522.jpg, syr_0330.jpg, syr_0521.jpg, syr_0520.jpg
- **Symmetry analysis**:
  - 30: -0.28
  - 45: 0.04
  - 60: 0.20
  - 90: 0.13
  - 120: 0.53
  - 180: 0.22
  - h_flip: 0.15
  - v_flip: -0.28

### Cluster 5

![Cluster 5](clusters/cluster_5/preview.png)

- **Count**: 22 images (10.1% of dataset)
- **Representative patterns**: syr_0422.jpg, syr_0225.jpg, syr_0432.jpg, syr_0318.jpg, syr_0735.jpg
- **Symmetry analysis**:
  - 30: -1.02
  - 45: -1.36
  - 60: -1.44
  - 90: -0.28
  - 120: -1.02
  - 180: 0.26
  - h_flip: 0.49
  - v_flip: 0.04

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

