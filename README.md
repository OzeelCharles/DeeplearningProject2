#  Chest X-Ray Pneumonia Classifier
*Deep Learning for Medical Imaging*

 🇬🇧 [**English Version**](#english-version) | 🇫🇷 [**Version Française**](#version-française)

---

<a id="english-version"></a>
## 🇬🇧 English Version

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%2F%20TensorFlow-FF6F00)
![Machine Learning](https://img.shields.io/badge/Task-Image%20Classification-success)

### Project Overview
This project focuses on the automatic detection of **Pneumonia** from chest X-ray images using Deep Learning. Medical imaging classification is highly sensitive, and the goal here is to build an AI assistant that can accurately flag potential cases to assist radiologists.

### The Journey & Methodology

Building a robust medical classifier is an iterative process. This project was developed in three distinct phases:

#### 1. The Baseline: Custom CNN (From Scratch)
I started by designing a Custom Convolutional Neural Network (CNN) from scratch. 
* **Goal:** Understand the spatial hierarchies of X-ray images and establish a performance baseline.
* **Architecture:** Sequential Convolutional layers followed by MaxPooling and fully connected layers.
* **Observation:** While the model learned basic patterns, it struggled with the subtle opacities typical of pneumonia and quickly showed signs of overfitting.

#### 2. Optimization Attempts
To improve the baseline, I implemented several optimization strategies:
* **Data Augmentation:** Rotation, zoom, and flipping to artificially increase the dataset size and improve generalization.
* **Regularization:** Added Dropout layers and Batch Normalization.
* **Hyperparameter Tuning:** Experimented with learning rate schedulers and different batch sizes.
* **Result:** The model became more robust and overfitting was reduced, but the overall accuracy and recall were still not at clinical standards.

#### 3. The Solution: Transfer Learning & Fine-Tuning
To reach state-of-the-art performance, I leveraged an external pre-trained model (**[Insert Model Name, e.g., ResNet50 / DenseNet121]**).
* **Approach:** I froze the base layers (which already knew how to extract complex visual features like edges and textures from millions of images) and replaced the classifier head with custom Dense layers.
* **Fine-Tuning:** Unfroze the top layers of the base model and trained the whole architecture with a very low learning rate.
* **Impact:** A massive leap in performance. The model successfully distinguished between healthy lungs and pneumonia with high **Recall/Sensitivity** (crucial in the medical field to avoid false negatives).

###  Results
* **Custom CNN Accuracy:** `[XX]%`
* **Fine-Tuned Model Accuracy:** `[XX]%`
* **Recall (Pneumonia Detection):** `[XX]%` *(Minimizing missed diagnoses)*

### How to run the project
```bash
# Clone the repository
git clone [https://github.com/your-username/pneumonia-classifier.git](https://github.com/your-username/pneumonia-classifier.git)
cd pneumonia-classifier

# Install dependencies
pip install -r requirements.txt

# Run the notebook or the prediction script
python predict.py --image path/to/xray.jpeg