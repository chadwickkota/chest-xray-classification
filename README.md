# chest-xray-classification
Deep Learning-Based COVID-19 &amp; Pneumonia Detection from Chest X-rays
# AI-Driven Chest X-Ray Classification for COVID-19 and Pneumonia Detection

This project leverages deep learning to automate the classification of chest X-ray images into three diagnostic categories: *COVID-19, **Pneumonia, and **Normal*. It aims to support radiologists with a fast, accurate, and interpretable tool, especially in resource-constrained settings.

**Watch Our Project Presentation on YouTube**: [Click here to watch](https://youtu.be/_ALUAdxjZF8)

##  Problem Statement

Diagnosing respiratory diseases through chest X-rays is difficult due to overlapping imaging features. Manual interpretation is time-consuming and varies across clinicians. Our objective is to:

- Develop CNN-based models for chest X-ray classification.
- Evaluate model performance on both internal and external datasets to test generalizability.
- Improve interpretability using Grad-CAM visualizations to support clinical decision-making.

## Models Implemented

- *SimpleCNN*: Custom 3-layer convolutional neural network with ReLU activations and Dropout (0.4), used as a baseline.
- *ResNet-50*: Pretrained model adapted for grayscale input, fine-tuned for classification.

*Loss Function*: Weighted CrossEntropy Loss to address class imbalance, particularly in underrepresented COVID-19 cases.

##  Datasets

- *Internal Dataset*: [COVID19-PNEUMONIA-NORMAL Chest X-ray Images](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chestxray-images) – 5,228 labeled images.Download here: https://drive.google.com/file/d/18ccLEZIadRzQD3yI7GB17cyHONhKRUZh/view?usp=sharing 
- *External Dataset*: [Chest X-ray COVID-19 & Pneumonia](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia) – 6,432 images split into training and test sets (20% test)..Download here: https://drive.google.com/file/d/10fe580BUYlIHl5qpEi3TG3uuBa06GBqc/view?usp=sharing 

These datasets are multi-source and expert-labeled, enhancing clinical relevance and diversity.

##  Workflow Overview

1. Load and preprocess internal and external datasets.
2. Apply noise-based augmentation:
   - Gaussian Noise
   - Speckle Noise
   - Random Contrast
3. Train SimpleCNN and ResNet-50 on the processed datasets.
4. Evaluate using Accuracy, Precision, Recall, and F1-score.
5. Generate Grad-CAM visualizations for model interpretability.

##  Results Summary

- *SimpleCNN* outperformed ResNet-50 on internal data with higher accuracy and recall.
- *ResNet-50* showed better generalization on external datasets but struggled with pneumonia recall.
- Grad-CAM visualizations helped reveal model focus areas and misclassifications.

##  Tools Used

- *PyTorch*: Model training and development  
- *Scikit-learn*: Performance evaluation  
- *Matplotlib*: Visualizations and Grad-CAM heatmaps

## Grad-CAM for Interpretability

Grad-CAM was applied to the final convolutional layer of ResNet-50 using forward and backward hooks. It visualizes important image regions influencing predictions, helping identify clinical relevance and model blind spots.

##  Future Enhancements

- Improve Grad-CAM alignment using expert-annotated datasets.
- Extend the approach to include CT scan or multi-modal image analysis.
- Implement advanced Explainable AI (XAI) frameworks (e.g., SHAP, LIME).
- Deploy the model into hospital systems (e.g., PACS/EHR) for real-time decision support.

##  References

1. Wang et al. (2020) – COVID-Net  
2. He et al. (2016) – ResNet  
3. Selvaraju et al. (2017) – Grad-CAM  
4. Albahli et al. (2021) – Noise Augmentation  
5. Zhou et al. (2016) – Discriminative Localization  
6. Kumar et al. (2022) – COVID-19 CXR Classification

##  Contributors

- *Rashmitha Eri*  
- *Chadwick Kota*

---

*Note*: This work was developed as part of a health informatics course project and is intended for educational and research purposes.
