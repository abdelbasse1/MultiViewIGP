# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 41 images from data/france
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 10

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 8
- **Silhouette Score**: 0.2301

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 2
- **Silhouette Score**: 0.2869

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.0756

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 9
- **Silhouette Score**: 0.1112

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 10
- **Silhouette Score**: 0.1509

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 9
- **Silhouette Score**: 0.1028

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 9
- **Silhouette Score**: 0.1525

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 9
- **Silhouette Score**: 0.1221

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 5
- **Silhouette Score**: 0.2656

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_Combined features show the highest agreement with the final consensus clustering (ARI = 0.77), while Texture features show the lowest agreement (ARI = -0.00).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 4 | 9.8% | 180 symmetry: 0.89, v_flip symmetry: 1.38 |
| 1 | 8 | 19.5% | No dominant characteristics |
| 2 | 7 | 17.1% | No dominant characteristics |
| 3 | 6 | 14.6% | 30 symmetry: 1.09, 45 symmetry: 0.79, 60 symmetry: 0.70 |
| 4 | 2 | 4.9% | 180 symmetry: 1.35, h_flip symmetry: 1.27 |
| 5 | 3 | 7.3% | v_flip symmetry: 1.54 |
| 6 | 4 | 9.8% | 60 symmetry: 0.94, 120 symmetry: 0.82 |
| 7 | 2 | 4.9% | 180 symmetry: 1.17 |
| 8 | 3 | 7.3% | No dominant characteristics |
| 9 | 2 | 4.9% | 30 symmetry: 1.46, 45 symmetry: 1.20, 60 symmetry: 1.18 |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 4 images (9.8% of dataset)
- **Representative patterns**: par_032.jpg, par_027.jpg, par_030.jpg, par_023.jpg
- **Symmetry analysis**:
  - 30: -0.69
  - 45: 0.01
  - 60: -0.28
  - 90: 0.51
  - 120: 0.42
  - 180: 0.89
  - h_flip: -0.58
  - v_flip: 1.38

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 8 images (19.5% of dataset)
- **Representative patterns**: par_015.jpg, par_016.jpg, par_017.jpg, par_019.jpg, par_018.jpg
- **Symmetry analysis**:
  - 30: -0.52
  - 45: -0.34
  - 60: -0.18
  - 90: 0.06
  - 120: 0.19
  - 180: 0.00
  - h_flip: -0.52
  - v_flip: -0.09

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 7 images (17.1% of dataset)
- **Representative patterns**: par_007.jpg, par_006.jpg, par_010.jpg, par_004.jpg, par_011.jpg
- **Symmetry analysis**:
  - 30: 0.09
  - 45: 0.01
  - 60: 0.09
  - 90: -0.23
  - 120: -0.04
  - 180: -0.49
  - h_flip: 0.65
  - v_flip: -0.88

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 6 images (14.6% of dataset)
- **Representative patterns**: par_038.jpg, par_039.jpg, par_034.jpg, par_035.jpg, par_037.jpg
- **Symmetry analysis**:
  - 30: 1.09
  - 45: 0.79
  - 60: 0.70
  - 90: -0.52
  - 120: -0.05
  - 180: -0.67
  - h_flip: -0.21
  - v_flip: 0.02

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 2 images (4.9% of dataset)
- **Representative patterns**: par_029.jpg, par_031.jpg
- **Symmetry analysis**:
  - 30: -0.85
  - 45: -1.28
  - 60: -1.60
  - 90: 0.10
  - 120: -1.57
  - 180: 1.35
  - h_flip: 1.27
  - v_flip: 0.15

### Cluster 5

![Cluster 5](clusters/cluster_5/preview.png)

- **Count**: 3 images (7.3% of dataset)
- **Representative patterns**: par_013.jpg, par_012.jpg, par_014.jpg
- **Symmetry analysis**:
  - 30: -0.11
  - 45: -0.71
  - 60: -0.89
  - 90: -0.17
  - 120: -0.91
  - 180: 0.32
  - h_flip: -0.92
  - v_flip: 1.54

### Cluster 6

![Cluster 6](clusters/cluster_6/preview.png)

- **Count**: 4 images (9.8% of dataset)
- **Representative patterns**: par_005.jpg, par_001.jpg, par_002.jpg, par_003.jpg
- **Symmetry analysis**:
  - 30: 0.65
  - 45: 0.69
  - 60: 0.94
  - 90: 0.31
  - 120: 0.82
  - 180: -0.41
  - h_flip: 0.47
  - v_flip: -0.56

### Cluster 7

![Cluster 7](clusters/cluster_7/preview.png)

- **Count**: 2 images (4.9% of dataset)
- **Representative patterns**: par_028.jpg, par_026.jpg
- **Symmetry analysis**:
  - 30: -1.37
  - 45: -1.27
  - 60: -1.32
  - 90: -0.65
  - 120: -1.33
  - 180: 1.17
  - h_flip: -0.24
  - v_flip: 0.19

### Cluster 8

![Cluster 8](clusters/cluster_8/preview.png)

- **Count**: 3 images (7.3% of dataset)
- **Representative patterns**: par_033.jpg, par_025.jpg, par_024.jpg
- **Symmetry analysis**:
  - 30: -0.33
  - 45: -0.04
  - 60: 0.06
  - 90: -0.07
  - 120: 0.69
  - 180: 0.47
  - h_flip: -0.33
  - v_flip: 0.08

### Cluster 9

![Cluster 9](clusters/cluster_9/preview.png)

- **Count**: 2 images (4.9% of dataset)
- **Representative patterns**: par_040.jpg, par_041.jpg
- **Symmetry analysis**:
  - 30: 1.46
  - 45: 1.20
  - 60: 1.18
  - 90: 1.40
  - 120: 0.26
  - 180: -0.97
  - h_flip: 1.46
  - v_flip: -0.99

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

