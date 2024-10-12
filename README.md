# 🔍 Bone Fracture Detection using Deep Learning

## 📝 Project Overview
This project, completed as part of my coursework, focuses on improving the accuracy of bone fracture diagnosis using radiographic images through advanced deep learning techniques. 
By leveraging neural network architectures such as Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), and Residual Networks (ResNet), the model classifies bone fractures from X-ray images. 
The project aims to assist medical professionals in diagnosing fractures more efficiently, addressing challenges related to misdiagnosis in complex and minor trauma cases.

## 🗂️ Dataset
The dataset is sourced from Kaggle's **Bone Fracture - Multi-Region X-ray** dataset, which contains 10,518 radiographic images divided into fractured and non-fractured categories. 
It covers multiple anatomical body regions, including lower limbs, upper limbs, hips, and knees. The dataset was preprocessed to ensure uniformity in image dimensions and quality.

- **Train Set**: 9,200 images
- **Validation Set**: 827 images
- **Test Set**: 491 images

## 🔬 Methodology
We employed several models to experiment with different levels of complexity:

1. **Image Preprocessing**:
    - Resized images to 224x224 pixels.
    - Converted images to grayscale for enhanced feature extraction.
    - Normalized pixel values to a range of 0 to 1.

2. **Deep Neural Network (DNN)**:
    - Three hidden layers with 128, 256, and 128 neurons, with dropout layers (rate 0.5).
    - Used sigmoid activation in the output layer for binary classification.
    - Achieved 83% accuracy on the test data.

3. **Convolutional Neural Network (CNN)**:
    - Two convolutional layers with 32 and 64 filters, followed by max-pooling.
    - Achieved 96.7% accuracy on the test data.

4. **ResNet Model**:
    - Utilized a pre-trained ResNet model (ImageNet weights) for feature extraction.
    - Achieved 48.4% accuracy.
    - Custom ResNet50 model improved accuracy to 92%.

## 📊 Results
| Model           | Test Accuracy |
|-----------------|---------------|
| DNN             | 83%           |
| CNN             | 96.7%         |
| Pre-trained ResNet | 48.4%         |
| Custom ResNet50 | 92%           |

## 🎯 Conclusion
While the CNN model provided the best accuracy at 96.7%, future improvements can be made by fine-tuning the ResNet model with more computational resources. The success of this project demonstrates the potential of deep learning in assisting radiologists in diagnosing bone fractures more efficiently.

## 🖥️ How to Run 

1. **Download the dataset**:
    - Visit the [Bone Fracture - Multi-Region X-ray Dataset on Kaggle](https://www.kaggle.com) and download the dataset. 
    - After downloading, extract the files and place them in a folder named `data` at the root level of this repository.

2. **Clone the repository**:
    - This command creates a local copy of the project repository on your machine:
    ```bash
    git clone https://github.com/mntsai/bone-fracture-detection-dl.git
    ```

3. **Install the required dependencies**:
    - Install the necessary Python libraries to run the code:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Jupyter notebooks**:
    - Choose and run one of the following notebooks:
        - `Final Project-DNN/CNN.ipynb` for the base deep learning model.
        - `Final Project-ResNet_Custom.ipynb` for the custom ResNet model.
        - `Final Project-ResNet_Pre_Trained.ipynb` for the pre-trained ResNet model.
        
    Open Jupyter Lab or Notebook and navigate to the corresponding file to execute:
    ```bash
    jupyter notebook Final\ Project-DNN/CNN.ipynb
    ```

5. **Configure dataset paths**:
    Ensure that the dataset path in the notebook is correctly pointing to the `data` folder where you extracted the dataset.
