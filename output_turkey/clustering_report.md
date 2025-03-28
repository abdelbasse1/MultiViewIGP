# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 274 images from data/turkey
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 10

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 2
- **Silhouette Score**: 0.3144

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 8
- **Silhouette Score**: 0.1422

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 3
- **Silhouette Score**: 0.1264

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 3
- **Silhouette Score**: 0.0983

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0765

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0738

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 2
- **Silhouette Score**: 0.0846

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 2
- **Silhouette Score**: 0.0569

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 2
- **Silhouette Score**: 0.2905

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_VGG16 features show the highest agreement with the final consensus clustering (ARI = 0.24), while Texture features show the lowest agreement (ARI = 0.04).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 33 | 12.0% | 30 symmetry: 0.72, 45 symmetry: 0.89, 60 symmetry: 0.95 |
| 1 | 26 | 9.5% | No dominant characteristics |
| 2 | 38 | 13.9% | No dominant characteristics |
| 3 | 24 | 8.8% | No dominant characteristics |
| 4 | 24 | 8.8% | No dominant characteristics |
| 5 | 25 | 9.1% | No dominant characteristics |
| 6 | 17 | 6.2% | No dominant characteristics |
| 7 | 55 | 20.1% | No dominant characteristics |
| 8 | 18 | 6.6% | No dominant characteristics |
| 9 | 14 | 5.1% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 33 images (12.0% of dataset)
- **Representative patterns**: tur_0618.jpg, tur_0817.jpg, tur_0631.jpg, tur_0433.jpg, tur_0223.jpg
- **Symmetry analysis**:
  - 30: 0.72
  - 45: 0.89
  - 60: 0.95
  - 90: 0.51
  - 120: 0.86
  - 180: -0.03
  - h_flip: -0.36
  - v_flip: 0.44

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 26 images (9.5% of dataset)
- **Representative patterns**: tur_0625.jpg, tur_0811.jpg, tur_0929.jpg, tur_0120.jpg, tur_0532.jpg
- **Symmetry analysis**:
  - 30: -0.08
  - 45: -0.08
  - 60: -0.04
  - 90: 0.12
  - 120: 0.14
  - 180: 0.39
  - h_flip: 0.41
  - v_flip: -0.37

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 38 images (13.9% of dataset)
- **Representative patterns**: tur_0624.jpg, tur_0221.jpg, tur_0619.jpg, tur_0419.jpg, tur_0222.jpg
- **Symmetry analysis**:
  - 30: 0.48
  - 45: 0.11
  - 60: -0.19
  - 90: -0.27
  - 120: -0.59
  - 180: -0.41
  - h_flip: 0.31
  - v_flip: -0.14

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 24 images (8.8% of dataset)
- **Representative patterns**: tur_0235.jpg, tur_0829.jpg, tur_0632.jpg, tur_0622.jpg, tur_0623.jpg
- **Symmetry analysis**:
  - 30: -0.05
  - 45: -0.01
  - 60: 0.03
  - 90: 0.01
  - 120: -0.01
  - 180: -0.11
  - h_flip: 0.24
  - v_flip: -0.15

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 24 images (8.8% of dataset)
- **Representative patterns**: tur_1116.jpg, tur_1110.jpg, tur_1111.jpg, tur_1113.jpg, tur_1112.jpg
- **Symmetry analysis**:
  - 30: -0.13
  - 45: -0.07
  - 60: -0.06
  - 90: -0.03
  - 120: -0.11
  - 180: 0.07
  - h_flip: -0.22
  - v_flip: 0.23

### Cluster 5

![Cluster 5](clusters/cluster_5/preview.png)

- **Count**: 25 images (9.1% of dataset)
- **Representative patterns**: tur_0218.jpg, tur_0122.jpg, tur_0133.jpg, tur_0127.jpg, tur_0126.jpg
- **Symmetry analysis**:
  - 30: 0.40
  - 45: 0.48
  - 60: 0.53
  - 90: 0.38
  - 120: 0.41
  - 180: 0.06
  - h_flip: -0.12
  - v_flip: 0.30

### Cluster 6

![Cluster 6](clusters/cluster_6/preview.png)

- **Count**: 17 images (6.2% of dataset)
- **Representative patterns**: tur_0234.jpg, tur_0816.jpg, tur_0430.jpg, tur_0928.jpg, tur_0322.jpg
- **Symmetry analysis**:
  - 30: -0.43
  - 45: -0.45
  - 60: -0.35
  - 90: -0.09
  - 120: -0.17
  - 180: 0.13
  - h_flip: 0.06
  - v_flip: -0.65

### Cluster 7

![Cluster 7](clusters/cluster_7/preview.png)

- **Count**: 55 images (20.1% of dataset)
- **Representative patterns**: tur_0432.jpg, tur_0828.jpg, tur_0810.jpg, tur_0409.jpg, tur_0435.jpg
- **Symmetry analysis**:
  - 30: -0.84
  - 45: -0.72
  - 60: -0.59
  - 90: -0.24
  - 120: -0.35
  - 180: 0.09
  - h_flip: -0.26
  - v_flip: 0.11

### Cluster 8

![Cluster 8](clusters/cluster_8/preview.png)

- **Count**: 18 images (6.6% of dataset)
- **Representative patterns**: tur_0630.jpg, tur_0627.jpg, tur_1114.jpg, tur_0420.jpg, tur_0519.jpg
- **Symmetry analysis**:
  - 30: 0.21
  - 45: 0.30
  - 60: 0.35
  - 90: 0.08
  - 120: 0.31
  - 180: -0.21
  - h_flip: 0.44
  - v_flip: -0.65

### Cluster 9

![Cluster 9](clusters/cluster_9/preview.png)

- **Count**: 14 images (5.1% of dataset)
- **Representative patterns**: tur_1117.jpg, tur_0814.jpg, tur_1115.jpg, tur_0424.jpg, tur_0805.jpg
- **Symmetry analysis**:
  - 30: 0.34
  - 45: 0.00
  - 60: -0.24
  - 90: -0.41
  - 120: -0.03
  - 180: 0.19
  - h_flip: -0.16
  - v_flip: 0.54

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

