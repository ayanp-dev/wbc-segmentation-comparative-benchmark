| Architecture / Model | Regime / Features | Test Accuracy (%) | Macro F1 | Sensitivity (Eosinophil) | Specificity (Eosinophil) | Latency (ms/img) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| SVM | Classical (Raw 51 Feats) | 96.25% | 0.9627 | 1.0000 | 1.0000 | 10.40 |
| PCA SVM | Classical (PCA 95%) | 94.25% | 0.9429 | 1.0000 | 1.0000 | 10.39 |
| Random Forest | Classical (Raw 51 Feats) | 94.25% | 0.9431 | 0.9900 | 0.9967 | 10.47 |
| PCA Random Forest | Classical (PCA 95%) | 93.75% | 0.9380 | 0.9600 | 0.9900 | 10.44 |
| ResNet-18 | Low-Data 15% (No Aug) | 96.00% | 0.9600 | 0.9900 | 1.0000 | 1.89 |
| ResNet-18 | Low-Data 15% + Domain Aug | 98.00% | 0.9800 | 0.9900 | 0.9967 | 1.90 |
| ResNet-18 | Full-Data 100% | 99.50% | 0.9950 | 1.0000 | 1.0000 | 1.90 |
