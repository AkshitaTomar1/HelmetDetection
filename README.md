# 🛡️ Helmet Detection Project

## 🚀 Project Overview
This project focuses on detecting whether riders are wearing helmets using deep learning and computer vision. 

## 📂 Project Structure
- **Helmet_Detection_notebook.ipynb**: Jupyter Notebook for model training, evaluation, and experimentation.
- **app.py**: Streamlit app for real-time image detection.
- **inference.py**: Helper functions (get_prediction, draw_boxes) for model inference and visualization.
- **model_weights.pth**: Trained model weights.
- **requirements.txt**: List of dependencies to recreate the environment.

## 🛠️ Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- PIL (Python Imaging Library)

## 🧠 Model Details
- **Model Architecture**: Faster R-CNN (ResNet-50-FPN backbone)
  
### Classes:
- rider-helmet-bike
  - **With Helmet**
  - **Without Helmet**

The model is fine-tuned for 3 classes and trained using custom helmet detection datasets.

