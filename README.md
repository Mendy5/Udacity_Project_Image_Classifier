# Udacity Introduction to Machine Learning with TensorFlow Project II: Create Your Own Image Classifier

This repository contains all related files of the Udacity Machine Learning with TensorFlow Project II Create Your Own Image Classifier

The project contains two Parts.
Part I: Developing an image clssifier model with Deep Learning (Tensor Flow)
Part II: Building the Command Line Application

## Installation
- python= 3.7.5
- tensorflow=2.4.0
- tf.keras=2.4.0
- argparse=1.1
- numpy=1.17.4

## Data Source
[OxFord Flowers 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

## Project Motivation
This is a trained classification model for a smart phone application. The users can get the name of the flowers by sending an image of the flower. 

## File Description
**Developing a Model**
This diretory contains all the related files for training the image classifier model

**Building an Application**
This direcotry contains all the scripts and files for the command line application

## Results
The image classification model is saved as image_classifier.h5. The final application is a command line application. 

How to apply the application?
To get the top 5 flowers that are the most likely the flowers on the image. Run the script on terminal
```bash
$python predict.py path_to_image image_classifier.h5
```
Ex:
```bash
$python predict.py ./test_images/wild_pansy.jpg image_classifier.h5
```
The results will return top 5 predicted classes and the probability of the predictions

To get top N flowers intead of top 5 predicted flowers. Run the script on terminal
```bash
$python predict.py path_to_image image_classifier.h5 --top_k K
```
Ex:
```bash
$python predict.py ./test_images/wild_pansy.jpg image_classifier.h5 --top_k 7
```
The results will return top 7 predicted classes and the probability of the predictions

The classes are number label in the dataset. To get a meaningful label of the flowers. We can parse a class and flower name mapping file into the model. Run the script on terminal
```bash
$python predict.py path_to_image image_classifier.h5 --category_names map.json
```
Ex:
```bash
$python predict.py ./test_images/wild_pansy.jpg image_classifier.h5 --category_names label_map.json
```

# Licensing, Authors, Acknowledgements
Must give credit to Maria-Elena Nilsback and Andrew Zisserman for the data.
