# 🛡️ Helmet Detection Project

## 🚀 Project Overview
This project focuses on detecting whether riders are wearing helmets using deep learning and computer vision. A Faster R-CNN model with a ResNet-50 FPN backbone is trained and deployed using Streamlit for an interactive web application.

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

## 📸 How to Run Locally

### Clone the Repository
```bash
git clone https://github.com/AkshitaTomar1/HelmetDetection.git
cd HelmetDetection
 
### Install Requirements
```bash
pip install -r requirements.txt

### Run the App
```bash
streamlit run app.py

Upload an Image
Upload an image and get predictions with bounding boxes and class labels.
