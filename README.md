# Reality Check
 _Reality Check_, an instance segementation detection model.


## Problem and Objective
With the latest innovations in deep fake technologies and video production, it has become more challenging to distinguish between real and AI-generated images and videos. As a consequence, malicious and harmful media can now be created and disseminated online with relative ease, which can cause significant harm to the mental health, reputation, and overall character of an individual. To address this issue, we created "Reality Check," an image detection model that utilizes instances of AI manipulation to accurately analyze and determine authentic and deepfaked images, rather than predictions on pure classification data alone.


## Methodologies
After downloading the dataset from Kaggle, we labeled instances of AI manipulation in the "fake images' folder and labeled instances of human facial proportions in the "real images" folder. We then uploaded the labeled dataset to Google Colab to begin pre-processing and training on the model. To ensure the maximum training, testing, and accuracy of the detection model, we utilized a 70-20-10 method to divide the labeled data. After training, the model we tested with the training and validation dataset, and a summary of the results can referenced below.

## Key Results
By our tests, our model can perform segmentation, analysis, and classification of images in as little as 2.0ms, with a 75-80% accuracy.

### Post Training Results
![Screenshot (84)](https://github.com/nmesosphere/Reality-Check/assets/65504077/887a4a1a-e5fe-450e-b3e8-020dd292996e)

## Data Sources
Dataset - https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

## Technologies Used
- Python
- Google Colaboratory
- Roboflow
- OpenCV
- Flask

## Authors
This project was developed by: 
- Mark-Anthony Delva ([ _@MrkAnthony_ ](https://github.com/MrkAnthony))
- Nmesoma Duru ([ _@nmesosphere_ ](https://github.com/nmesosphere))
- Micheal Johnson ([ _@SolaMike_ ](https://github.com/SolaMike))

