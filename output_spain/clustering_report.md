# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 452 images from data/spain
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 10

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 4
- **Silhouette Score**: 0.2301

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 2
- **Silhouette Score**: 0.1535

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.2440

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 2
- **Silhouette Score**: 0.1006

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.1228

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0397

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 2
- **Silhouette Score**: 0.0801

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 2
- **Silhouette Score**: 0.0965

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 2
- **Silhouette Score**: 0.2917

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_MobileNet features show the highest agreement with the final consensus clustering (ARI = 0.21), while Texture features show the lowest agreement (ARI = 0.01).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 25 | 5.5% | No dominant characteristics |
| 1 | 38 | 8.4% | No dominant characteristics |
| 2 | 81 | 17.9% | No dominant characteristics |
| 3 | 44 | 9.7% | 30 symmetry: 0.96, 45 symmetry: 1.14, 60 symmetry: 1.15 |
| 4 | 11 | 2.4% | h_flip symmetry: 0.71 |
| 5 | 78 | 17.3% | No dominant characteristics |
| 6 | 42 | 9.3% | 30 symmetry: 1.03 |
| 7 | 54 | 11.9% | No dominant characteristics |
| 8 | 42 | 9.3% | No dominant characteristics |
| 9 | 37 | 8.2% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 25 images (5.5% of dataset)
- **Representative patterns**: spa_1623x.jpg, spa_2002x.jpg, spa_2110x.jpg, spa_2009x.jpg, spa_2013x.jpg
- **Symmetry analysis**:
  - 30: -0.21
  - 45: -0.42
  - 60: -0.48
  - 90: -0.09
  - 120: -0.46
  - 180: 0.05
  - h_flip: 0.17
  - v_flip: 0.31

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 38 images (8.4% of dataset)
- **Representative patterns**: spa_2015x.jpg, spa_2107x.jpg, spa_2111x.jpg, spa_0423x.jpg, spa_2023x.jpg
- **Symmetry analysis**:
  - 30: -0.13
  - 45: -0.40
  - 60: -0.47
  - 90: -0.08
  - 120: -0.61
  - 180: -0.24
  - h_flip: 0.66
  - v_flip: -0.39

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 81 images (17.9% of dataset)
- **Representative patterns**: spa_0202.jpg, spa_0606.jpg, spa_2207.jpg, spa_2206.jpg, spa_2212.jpg
- **Symmetry analysis**:
  - 30: -0.33
  - 45: -0.12
  - 60: 0.01
  - 90: -0.00
  - 120: 0.40
  - 180: 0.14
  - h_flip: -0.19
  - v_flip: -0.11

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 44 images (9.7% of dataset)
- **Representative patterns**: spa_2830.jpg, spa_1914.jpg, spa_0415x.jpg, spa_0507x.jpg, spa_0511x.jpg
- **Symmetry analysis**:
  - 30: 0.96
  - 45: 1.14
  - 60: 1.15
  - 90: 0.50
  - 120: 1.06
  - 180: 0.19
  - h_flip: 0.35
  - v_flip: 0.01

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 11 images (2.4% of dataset)
- **Representative patterns**: spa_2832.jpg, spa_1913.jpg, spa_0102.jpg, spa_0114.jpg, spa_0100.jpg
- **Symmetry analysis**:
  - 30: -0.21
  - 45: -0.30
  - 60: -0.31
  - 90: 0.13
  - 120: -0.46
  - 180: 0.03
  - h_flip: 0.71
  - v_flip: 0.19

### Cluster 5

![Cluster 5](clusters/cluster_5/preview.png)

- **Count**: 78 images (17.3% of dataset)
- **Representative patterns**: spa_1615x.jpg, spa_0907x.jpg, spa_0931x.jpg, spa_1531x.jpg, spa_2822.jpg
- **Symmetry analysis**:
  - 30: -0.58
  - 45: -0.55
  - 60: -0.53
  - 90: -0.26
  - 120: -0.43
  - 180: 0.03
  - h_flip: -0.21
  - v_flip: 0.14

### Cluster 6

![Cluster 6](clusters/cluster_6/preview.png)

- **Count**: 42 images (9.3% of dataset)
- **Representative patterns**: spa_2824.jpg, spa_1915.jpg, spa_2831.jpg, spa_2825.jpg, spa_0217.jpg
- **Symmetry analysis**:
  - 30: 1.03
  - 45: 0.65
  - 60: 0.41
  - 90: 0.22
  - 120: -0.27
  - 180: -0.19
  - h_flip: -0.01
  - v_flip: 0.18

### Cluster 7

![Cluster 7](clusters/cluster_7/preview.png)

- **Count**: 54 images (11.9% of dataset)
- **Representative patterns**: spa_2818.jpg, spa_0821.jpg, spa_0809.jpg, spa_0613.jpg, spa_2210.jpg
- **Symmetry analysis**:
  - 30: -0.38
  - 45: -0.25
  - 60: -0.16
  - 90: -0.15
  - 120: -0.00
  - 180: 0.05
  - h_flip: -0.06
  - v_flip: -0.19

### Cluster 8

![Cluster 8](clusters/cluster_8/preview.png)

- **Count**: 42 images (9.3% of dataset)
- **Representative patterns**: spa_0820.jpg, spa_1642x.jpg, spa_1109.jpg, spa_1321.jpg, spa_1912.jpg
- **Symmetry analysis**:
  - 30: 0.15
  - 45: 0.08
  - 60: 0.00
  - 90: -0.22
  - 120: -0.20
  - 180: -0.31
  - h_flip: -0.11
  - v_flip: -0.18

### Cluster 9

![Cluster 9](clusters/cluster_9/preview.png)

- **Count**: 37 images (8.2% of dataset)
- **Representative patterns**: spa_2819.jpg, spa_2211.jpg, spa_2826.jpg, spa_0214.jpg, spa_1906.jpg
- **Symmetry analysis**:
  - 30: 0.37
  - 45: 0.40
  - 60: 0.40
  - 90: 0.29
  - 120: 0.37
  - 180: 0.09
  - h_flip: -0.32
  - v_flip: 0.36

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

