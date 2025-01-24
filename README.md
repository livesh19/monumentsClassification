# Monument Classification Using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of famous monuments from around the world. The model is trained using a dataset containing images of 12 different monuments and is capable of identifying the monument in any given image.

---


## About the Project

The project uses TensorFlow and Keras to implement and train a CNN model to classify the following monuments:
- Burj Khalifa
- Chichen Itza
- Christ the Redeemer
- Eiffel Tower
- Great Wall of China
- Machu Picchu
- Pyramids of Giza
- Roman Colosseum
- Statue of Liberty
- Stonehenge
- Taj Mahal
- Venezuela Angel Falls

The model uses augmented training data to ensure robust performance in recognizing images from various angles and lighting conditions.

---

## Dataset

The dataset contains a total of **3846 images**, organized into subfolders for each monument. The dataset structure is as follows:

## Model Architecture

The model architecture consists of:
1. **Three Convolutional Layers** with ReLU activation and MaxPooling.
2. **Flattening Layer** to convert feature maps into a vector.
3. **Fully Connected Layer** with 128 neurons and ReLU activation.
4. **Output Layer** with `softmax` activation for multi-class classification.

The model uses **categorical cross-entropy loss** and the **Adam optimizer**.

---

## Requirements

Make sure you have the following installed:
- Python 3.7+
- TensorFlow 2.x
- Matplotlib
- NumPy

You can install the required Python packages using:
```bash
pip install -r requirements.txt


Clone the repository:

git clone https://github.com/<your-username>/monument-classification.git
Navigate to the project directory:

cd monument-classification
Install the dependencies:

pip install -r requirements.txt
