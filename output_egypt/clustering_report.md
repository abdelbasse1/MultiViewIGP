# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 449 images from data/egypt
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 10

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 5
- **Silhouette Score**: 0.1623

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 2
- **Silhouette Score**: 0.1153

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.1702

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 3
- **Silhouette Score**: 0.0568

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 7
- **Silhouette Score**: 0.1006

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 5
- **Silhouette Score**: 0.0509

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 8
- **Silhouette Score**: 0.1137

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 10
- **Silhouette Score**: 0.0709

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 2
- **Silhouette Score**: 0.2985

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_Combined features show the highest agreement with the final consensus clustering (ARI = 0.65), while Symmetry features show the lowest agreement (ARI = 0.03).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 30 | 6.7% | No dominant characteristics |
| 1 | 32 | 7.1% | 30 symmetry: 0.83, 45 symmetry: 0.97, 60 symmetry: 1.21 |
| 2 | 87 | 19.4% | No dominant characteristics |
| 3 | 56 | 12.5% | No dominant characteristics |
| 4 | 14 | 3.1% | No dominant characteristics |
| 5 | 28 | 6.2% | 180 symmetry: 1.39, h_flip symmetry: 0.76, v_flip symmetry: 1.37 |
| 6 | 64 | 14.3% | No dominant characteristics |
| 7 | 13 | 2.9% | No dominant characteristics |
| 8 | 95 | 21.2% | No dominant characteristics |
| 9 | 30 | 6.7% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 30 images (6.7% of dataset)
- **Representative patterns**: egy_1716.jpg, egy_1528.jpg, egy_1515.jpg, egy_1529.jpg, egy_1715.jpg
- **Symmetry analysis**:
  - 30: -0.42
  - 45: -0.29
  - 60: -0.07
  - 90: -0.03
  - 120: 0.23
  - 180: 0.07
  - h_flip: -0.51
  - v_flip: -0.29

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 32 images (7.1% of dataset)
- **Representative patterns**: egy_0309x.jpg, egy_0221x.jpg, egy_0333x.jpg, egy_0313x.jpg, egy_0411x.jpg
- **Symmetry analysis**:
  - 30: 0.83
  - 45: 0.97
  - 60: 1.21
  - 90: 1.52
  - 120: 1.43
  - 180: 0.88
  - h_flip: 0.49
  - v_flip: 0.28

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 87 images (19.4% of dataset)
- **Representative patterns**: egy_0636.jpg, egy_1110.jpg, egy_1104.jpg, egy_1105.jpg, egy_1111.jpg
- **Symmetry analysis**:
  - 30: -0.24
  - 45: -0.35
  - 60: -0.39
  - 90: -0.20
  - 120: -0.39
  - 180: -0.15
  - h_flip: 0.08
  - v_flip: -0.07

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 56 images (12.5% of dataset)
- **Representative patterns**: egy_0431x.jpg, egy_0427x.jpg, egy_0325x.jpg, egy_0305x.jpg, egy_0217x.jpg
- **Symmetry analysis**:
  - 30: 0.48
  - 45: 0.61
  - 60: 0.59
  - 90: 0.46
  - 120: 0.61
  - 180: 0.26
  - h_flip: 0.11
  - v_flip: -0.41

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 14 images (3.1% of dataset)
- **Representative patterns**: egy_1312.jpg, egy_1313.jpg, egy_0812.jpg, egy_1311.jpg, egy_1310.jpg
- **Symmetry analysis**:
  - 30: 0.38
  - 45: 0.18
  - 60: -0.04
  - 90: -0.58
  - 120: -0.45
  - 180: -0.69
  - h_flip: -0.48
  - v_flip: -0.29

### Cluster 5

![Cluster 5](clusters/cluster_5/preview.png)

- **Count**: 28 images (6.2% of dataset)
- **Representative patterns**: egy_0805.jpg, egy_1306.jpg, egy_1307.jpg, egy_1501.jpg, egy_0804.jpg
- **Symmetry analysis**:
  - 30: -0.66
  - 45: -1.19
  - 60: -1.37
  - 90: -0.45
  - 120: -0.88
  - 180: 1.39
  - h_flip: 0.76
  - v_flip: 1.37

### Cluster 6

![Cluster 6](clusters/cluster_6/preview.png)

- **Count**: 64 images (14.3% of dataset)
- **Representative patterns**: egy_1517.jpg, egy_1512.jpg, egy_0618.jpg, egy_0619.jpg, egy_1513.jpg
- **Symmetry analysis**:
  - 30: -0.14
  - 45: -0.11
  - 60: -0.06
  - 90: -0.19
  - 120: -0.06
  - 180: -0.17
  - h_flip: 0.38
  - v_flip: -0.56

### Cluster 7

![Cluster 7](clusters/cluster_7/preview.png)

- **Count**: 13 images (2.9% of dataset)
- **Representative patterns**: egy_0811.jpg, egy_1305.jpg, egy_1304.jpg, egy_1315.jpg, egy_0633.jpg
- **Symmetry analysis**:
  - 30: 0.19
  - 45: 0.01
  - 60: -0.10
  - 90: -0.18
  - 120: -1.20
  - 180: -1.18
  - h_flip: -0.58
  - v_flip: -0.04

### Cluster 8

![Cluster 8](clusters/cluster_8/preview.png)

- **Count**: 95 images (21.2% of dataset)
- **Representative patterns**: egy_1514.jpg, lou_080.jpg, lou_081.jpg, egy_0201x.jpg, egy_1729.jpg
- **Symmetry analysis**:
  - 30: 0.07
  - 45: 0.16
  - 60: 0.14
  - 90: -0.11
  - 120: 0.01
  - 180: -0.29
  - h_flip: -0.51
  - v_flip: 0.49

### Cluster 9

![Cluster 9](clusters/cluster_9/preview.png)

- **Count**: 30 images (6.7% of dataset)
- **Representative patterns**: egy_1702.jpg, egy_1703.jpg, egy_1701.jpg, egy_0609.jpg, egy_0608.jpg
- **Symmetry analysis**:
  - 30: -0.22
  - 45: -0.10
  - 60: -0.19
  - 90: -0.34
  - 120: -0.13
  - 180: -0.23
  - h_flip: 0.14
  - v_flip: -0.51

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

