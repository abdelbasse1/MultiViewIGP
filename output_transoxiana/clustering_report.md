# Multiview Clustering Analysis of Islamic Geometric Patterns

## 1. Overview

- **Dataset**: 303 images from data/transoxiana
- **Clustering Method**: Multiview consensus clustering
- **Feature Types**: Color, Texture, Deep_VGG16, Deep_ResNet50, Deep_EfficientNet, Deep_MobileNet, Deep_DenseNet, Deep_Combined, Symmetry
- **Number of Clusters**: 7

## 2. Feature Analysis

### Color Features

- **Dimension**: 96
- **Optimal k**: 2
- **Silhouette Score**: 0.1159

![Color Metrics](metrics_Color.png)

![Color t-SNE](tsne_Color.png)

### Texture Features

- **Dimension**: 1577
- **Optimal k**: 3
- **Silhouette Score**: 0.0595

![Texture Metrics](metrics_Texture.png)

![Texture t-SNE](tsne_Texture.png)

### Deep_VGG16 Features

- **Dimension**: 512
- **Optimal k**: 2
- **Silhouette Score**: 0.1264

![Deep_VGG16 Metrics](metrics_Deep_VGG16.png)

![Deep_VGG16 t-SNE](tsne_Deep_VGG16.png)

### Deep_ResNet50 Features

- **Dimension**: 2048
- **Optimal k**: 2
- **Silhouette Score**: 0.0882

![Deep_ResNet50 Metrics](metrics_Deep_ResNet50.png)

![Deep_ResNet50 t-SNE](tsne_Deep_ResNet50.png)

### Deep_EfficientNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.0561

![Deep_EfficientNet Metrics](metrics_Deep_EfficientNet.png)

![Deep_EfficientNet t-SNE](tsne_Deep_EfficientNet.png)

### Deep_MobileNet Features

- **Dimension**: 1280
- **Optimal k**: 2
- **Silhouette Score**: 0.1305

![Deep_MobileNet Metrics](metrics_Deep_MobileNet.png)

![Deep_MobileNet t-SNE](tsne_Deep_MobileNet.png)

### Deep_DenseNet Features

- **Dimension**: 1024
- **Optimal k**: 2
- **Silhouette Score**: 0.0622

![Deep_DenseNet Metrics](metrics_Deep_DenseNet.png)

![Deep_DenseNet t-SNE](tsne_Deep_DenseNet.png)

### Deep_Combined Features

- **Dimension**: 6144
- **Optimal k**: 2
- **Silhouette Score**: 0.0367

![Deep_Combined Metrics](metrics_Deep_Combined.png)

![Deep_Combined t-SNE](tsne_Deep_Combined.png)

### Symmetry Features

- **Dimension**: 8
- **Optimal k**: 2
- **Silhouette Score**: 0.3300

![Symmetry Metrics](metrics_Symmetry.png)

![Symmetry t-SNE](tsne_Symmetry.png)

## 3. Consensus Matrix

The consensus matrix represents the agreement between different feature views on which pairs of patterns should be clustered together.

![Consensus Matrix](consensus_matrix.png)

## 3.5. Agreement Between Views (ARI)

The Adjusted Rand Index (ARI) measures the agreement between different clustering views and the final consensus clustering.

![ARI Comparison](ari_comparison.png)

The Deep_Combined features show the highest agreement with the final consensus clustering (ARI = 0.23), while Color features show the lowest agreement (ARI = 0.02).

## 4. Clustering Results

![Cluster Overview](cluster_overview.png)

### Cluster Summary

| Cluster | Count | % of Dataset | Key Characteristics |
|---------|-------|-------------|---------------------|
| 0 | 64 | 21.1% | No dominant characteristics |
| 1 | 28 | 9.2% | 30 symmetry: 1.38, 45 symmetry: 1.40, 60 symmetry: 1.28 |
| 2 | 40 | 13.2% | No dominant characteristics |
| 3 | 56 | 18.5% | No dominant characteristics |
| 4 | 22 | 7.3% | 45 symmetry: 0.82, 60 symmetry: 0.88, 120 symmetry: 0.74 |
| 5 | 45 | 14.9% | No dominant characteristics |
| 6 | 48 | 15.8% | No dominant characteristics |


## 5. Detailed Cluster Analysis

### Cluster 0

![Cluster 0](clusters/cluster_0/preview.png)

- **Count**: 64 images (21.1% of dataset)
- **Representative patterns**: tra_0811.jpg, tra_0623.jpg, tra_0806.jpg, tra_0423.jpg, 7pzvwd30Lf5gRzXG12i4397mDyr1UdPK2TWG9nPx.jpg
- **Symmetry analysis**:
  - 30: 0.15
  - 45: 0.09
  - 60: 0.07
  - 90: -0.08
  - 120: 0.03
  - 180: -0.06
  - h_flip: -0.07
  - v_flip: 0.27

### Cluster 1

![Cluster 1](clusters/cluster_1/preview.png)

- **Count**: 28 images (9.2% of dataset)
- **Representative patterns**: tra_0227.jpg, tra_0218.jpg, tra_0618.jpg, tra_0624.jpg, tra_0803.jpg
- **Symmetry analysis**:
  - 30: 1.38
  - 45: 1.40
  - 60: 1.28
  - 90: 0.76
  - 120: 0.71
  - 180: 0.01
  - h_flip: 0.31
  - v_flip: -0.14

### Cluster 2

![Cluster 2](clusters/cluster_2/preview.png)

- **Count**: 40 images (13.2% of dataset)
- **Representative patterns**: tra_1111.jpg, tra_0421.jpg, tra_0810.jpg, tra_0231.jpg, tra_0634.jpg
- **Symmetry analysis**:
  - 30: -0.49
  - 45: -0.64
  - 60: -0.66
  - 90: -0.33
  - 120: -0.49
  - 180: 0.15
  - h_flip: 0.16
  - v_flip: -0.01

### Cluster 3

![Cluster 3](clusters/cluster_3/preview.png)

- **Count**: 56 images (18.5% of dataset)
- **Representative patterns**: tra_0420.jpg, tra_0409.jpg, 2lVsDg3qe6lrfBB5NlZJ159krmJI262Y2oDrJmiE.jpg, tra_0609.jpg, tra_0812.jpg
- **Symmetry analysis**:
  - 30: -0.50
  - 45: -0.35
  - 60: -0.29
  - 90: -0.24
  - 120: -0.04
  - 180: -0.04
  - h_flip: -0.30
  - v_flip: -0.19

### Cluster 4

![Cluster 4](clusters/cluster_4/preview.png)

- **Count**: 22 images (7.3% of dataset)
- **Representative patterns**: tra_0408.jpg, tra_0425.jpg, tra_0236.jpg, tra_0418.jpg, tra_0324.jpg
- **Symmetry analysis**:
  - 30: 0.61
  - 45: 0.82
  - 60: 0.88
  - 90: 0.34
  - 120: 0.74
  - 180: -0.15
  - h_flip: -0.13
  - v_flip: -0.19

### Cluster 5

![Cluster 5](clusters/cluster_5/preview.png)

- **Count**: 45 images (14.9% of dataset)
- **Representative patterns**: tra_0226.jpg, tra_0232.jpg, tra_0233.jpg, tra_0804.jpg, tra_0621.jpg
- **Symmetry analysis**:
  - 30: -0.02
  - 45: -0.22
  - 60: -0.28
  - 90: 0.06
  - 120: -0.52
  - 180: -0.14
  - h_flip: 0.51
  - v_flip: -0.49

### Cluster 6

![Cluster 6](clusters/cluster_6/preview.png)

- **Count**: 48 images (15.8% of dataset)
- **Representative patterns**: tra_0622.jpg, tra_0805.jpg, tra_0434.jpg, tra_0224.jpg, tra_0620.jpg
- **Symmetry analysis**:
  - 30: -0.28
  - 45: -0.18
  - 60: -0.10
  - 90: 0.00
  - 120: 0.16
  - 180: 0.19
  - h_flip: -0.28
  - v_flip: 0.50

## 6. Conclusion

This analysis demonstrates the effectiveness of multiview clustering for categorizing Islamic geometric patterns. By combining color, texture, deep learning features, and symmetry analysis, we've identified distinctive pattern clusters with specific geometric and visual characteristics.

These results can provide valuable insights for researchers studying Islamic art and architecture, enabling quantitative classification of pattern styles based on their mathematical and visual properties.

