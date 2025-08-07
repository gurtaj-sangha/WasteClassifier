# WasteClassifier

This project uses transfer learning with ResNet50 to classify images into three categories: garbage, recycling, and compost.

## 📁 Dataset Structure
Make sure your dataset is structured like this (as expected by `torchvision.datasets.ImageFolder`):
```
Dataset/
├── garbage/
├── recycling/
└── compost/
```

##  How to Train the Model
Run the training script:
```bash
python model.py
```
It will train the model using 3 different learning rates (0.1, 0.001, 0.0001), and save the best one for testing.

##  Evaluation
The script will automatically generate:
- Accuracy/loss graphs for training and validation
- A confusion matrix for the test set

##  Running the Streamlit Interface
If you have a Streamlit app, run it like this:
```bash
streamlit run app.py
```

##  Setup Instructions
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have your dataset placed in a folder named `Dataset`.

3. Optional: Create a virtual environment for isolation:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---
Built by The Decompilers - Gurtaj Sangha, Ricky Tat, Michelle Park
