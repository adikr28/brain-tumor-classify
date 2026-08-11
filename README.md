# 🧠 Brain Tumor MRI Classification

A deep learning project for classifying brain MRI images into four categories using **Transfer Learning with ResNet18** and **PyTorch**.

> ⚠️ **Disclaimer:** This project is an educational/research prototype and is **not a medical diagnostic system**. Predictions should not be used for clinical decision-making.

---

## 📌 Project Overview

Brain tumors can appear in different forms in MRI scans. This project explores the use of deep learning to automatically classify MRI images into four categories:

* **Glioma**
* **Meningioma**
* **No Tumor**
* **Pituitary Tumor**

The model uses a pretrained **ResNet18** convolutional neural network and adapts its final classification layer to predict the four target classes.

### Workflow

```text
MRI Dataset
     ↓
Image Preprocessing
     ↓
Resize to 224 × 224
     ↓
Tensor Conversion & Normalization
     ↓
Pretrained ResNet18
     ↓
Transfer Learning
     ↓
4-Class Classification
     ↓
Prediction & Evaluation
```

---

## 🚀 Features

* MRI image classification
* Four-class tumor classification
* Transfer learning using ResNet18
* GPU/CPU support through PyTorch
* Image preprocessing and normalization
* Model training and evaluation
* Confusion matrix generation
* Individual MRI image prediction
* Batch prediction and CSV export
* Saved PyTorch model (`.pth`)

---

## 🛠️ Tech Stack

| Technology   | Purpose                                             |
| ------------ | --------------------------------------------------- |
| Python       | Programming language                                |
| PyTorch      | Deep learning framework                             |
| Torchvision  | Dataset utilities, transforms and pretrained models |
| ResNet18     | Image classification model                          |
| Pandas       | Prediction/result processing                        |
| NumPy        | Numerical operations                                |
| Matplotlib   | Visualization                                       |
| Scikit-learn | Evaluation and confusion matrix                     |
| Google Colab | Model training environment                          |
| GitHub       | Version control and project hosting                 |

---

## 📂 Dataset Structure

The notebook uses an `ImageFolder`-style dataset with separate training and testing directories.

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Each class directory contains MRI images belonging to that category.

---

## 🔄 Image Preprocessing

The images are processed before being passed to the neural network.

The main preprocessing pipeline used in the notebook is:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

Some augmentation experiments were also included in the notebook, such as:

* Random horizontal flipping
* Random rotation
* Brightness adjustment
* Contrast adjustment

These experiments will be cleaned and properly incorporated into the final version of the project.

---

## 🤖 Model

### ResNet18

The project uses **ResNet18**, a convolutional neural network pretrained on ImageNet.

Instead of training the entire network from scratch, the project initially freezes the pretrained layers and replaces the final fully connected layer:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)
```

The final model predicts four classes:

```text
0 → Glioma
1 → Meningioma
2 → No Tumor
3 → Pituitary
```

### Why Transfer Learning?

Transfer learning allows a pretrained model to reuse visual features learned from a large image dataset and adapt them to a new classification task.

This can reduce:

* Training time
* Required computational resources
* Amount of training needed from scratch

---

## ⚙️ Training

The initial implementation uses:

```text
Model       : ResNet18
Optimizer   : Adam
Loss        : CrossEntropyLoss
Learning Rate: 0.001
Batch Size  : 32
Epochs      : 10
```

The model automatically uses a GPU when one is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 📊 Evaluation

The notebook includes several evaluation approaches:

* Test accuracy
* Confusion matrix
* Individual image predictions
* Batch predictions
* Prediction CSV generation

Example prediction output:

```text
Image              Actual       Prediction
------------------------------------------------
Te-gl_0010.jpg     glioma       glioma
Te-me_0012.jpg     meningioma   meningioma
```

A confusion matrix is also generated to analyze how the model performs across the four classes.

---

## 💾 Model

The trained model is saved using PyTorch:

```python
torch.save(
    model.state_dict(),
    "brain_tumor_resnet18.pth"
)
```

The saved model can subsequently be loaded for inference without retraining.

---

## 🧪 Making a Prediction

After loading the trained model, an MRI image can be processed and passed through the model:

```python
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)

print(classes[pred.item()])
```

The model returns one of the four supported classes.

---

## 📈 Results

**Final performance metrics will be added after the notebook is cleaned and the model is re-evaluated using a proper train/validation/test methodology.**

The current notebook contains several experimental evaluation sections and manually specified/example values. Therefore, those values should **not yet be treated as the official final model performance**.

Final results will include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🌐 Planned Web Application

The next stage of this project is to convert the trained model into a functional web application.

### Planned architecture

```text
                 ┌──────────────────┐
                 │   User Uploads   │
                 │    MRI Image     │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │    Web Frontend  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   Backend API    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   ResNet18 Model │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   Prediction     │
                 │ + Confidence     │
                 └──────────────────┘
```

The website will allow users to upload an MRI image and receive the model's predicted class.

The web application will be presented as an **educational AI demonstration**, not as a clinical diagnostic tool.

---

## 📁 Planned Project Structure

```text
brain-tumor-classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Brain_tumor.ipynb
│
├── model/
│   └── brain_tumor_resnet18.pth
│
├── backend/
│   └── ...
│
├── frontend/
│   └── ...
│
├── data/
│   └── README.md
│
└── .gitignore
```

Large datasets will not be committed directly to the repository.

---

## 🔮 Future Improvements

* Proper train/validation/test split
* Data augmentation
* Hyperparameter tuning
* Precision, recall and F1-score
* Better class-balance analysis
* Grad-CAM / model explainability
* Confidence scores
* Improved inference pipeline
* Functional web interface
* Backend API
* Model deployment
* Containerization
* Better error handling and input validation

---

## ⚠️ Limitations

This project has several limitations that will be addressed during the rebuild:

1. The original notebook contains duplicated experimental code.
2. Some evaluation sections use the testing dataset during model development.
3. Some visualizations contain manually specified/example values.
4. The final reported accuracy has not yet been independently verified.
5. The model has not been clinically validated.
6. The model should not be used for medical diagnosis.

These issues will be addressed before considering the project a final production-ready implementation.

---

## 👨‍💻 Author

**Aditya Kumar**

Computer Science Engineering Student
Interested in Data Analytics, Machine Learning and Software Development.

GitHub: [@adikr28](https://github.com/adikr28)

---

## ⭐ Project Status

**Current:** 🟡 Model prototype completed

**Next:** 🔧 Clean and validate the ML pipeline

**Then:** 🌐 Build and deploy the functional web application
