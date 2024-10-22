# Brand Recognition System / Fresh Produce Freshness Detection using CNN


## Project Overview
This project implements a convolutional neural network (CNN) to detect the freshness of fresh produce based on images. It involves building a full-stack application with a Python-based backend and a front-end developed using HTML, CSS, and JavaScript. The freshness detection model is trained on a dataset from Kaggle, and real-time predictions are served via a Flask API.

## Technologies Used
- **Algorithm**: Convolutional Neural Network (CNN)
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask Framework)
- **Database**: SQLite
- **Dataset**: Kaggle
- **Development Tools**: 
  - Visual Studio Code (VSCode)
  - Jupyter Notebook

## Implementation Steps
1. **Dataset Preparation**: 
   - Download and preprocess the dataset from Kaggle, ensuring the data is suitable for training a CNN model.

2. **CNN Model Training**: 
   - Develop and train a CNN model using Python in Jupyter Notebook. This model will be used for detecting the freshness of produce based on image input.

3. **Frontend Development**: 
   - Design the user interface using HTML, CSS, and JavaScript, where users can upload images for freshness detection.

4. **Backend Development with Flask**: 
   - Build the Flask backend to handle image uploads, pass the images to the trained CNN model for prediction, and return the results to the frontend.

5. **Real-Time Testing on Trained Model**: 
   - Integrate the trained model with the Flask backend and test its performance using real-time inputs.

6. **Prediction Results**: 
   - Display the freshness prediction results on the frontend after processing the input images.

## Installation

### Prerequisites
- Python 3.x
- Flask
- SQLite
- Virtual environment tools (optional but recommended)

## Libraries required
-flask==1.1.1
-werkzeug==0.15.6 
-itsdangerous==2.0.1 
-jinja2==3.0.3 
-opencv-python==4.5.3.56 
-tensorflow==2.4.0 
-keras==2.4.3 
-pillow==8.1.0 
-imutils==0.5.4 
-pandas==1.2.1 
-matplotlib==3.3.4 
-protobuf==3.19.0 
-numpy==1.19.5 
-scikit-learn==0.24.1

