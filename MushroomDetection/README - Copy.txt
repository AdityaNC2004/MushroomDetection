# CAPSTONE PROJECT - Mushroom Identification & Report App


## Introduction
This project is designed to create a functioning mushroom identification tool, looking to identify the species of a mushroom from a given image and hand back a report about the mushroom, including features such as its edibility, environmental data from samples taken of the mushroom species, and spottings of that species around the world by mushroom enthusiasts. This mushroom identification app can recognize 100 species of mushrooms, and aims to provide information to the users about the mushroom they have spotted.

## Project Overview
The source data for this project was provided by a [kaggle dataset](https://www.kaggle.com/datasets/derekkunowilliams/mushrooms) which contains images of mushrooms, grouped in folders by species and classified by level of edibility, from deadly to poisonous, contitionally edible, or edible. The python code for this project narrows down the species to 100 species with the most image data with equal halfs of edible and non edible species, using the [OS](https://docs.python.org/3/library/os.html) library in conjunction with [Pandas](https://pandas.pydata.org/). The chosen folders are then supplemented by more images of mushrooms for each species taken from the Internet and saved into the respective folders using a [zip file extension](https://download-all-images.mobilefirst.me/). The pictures in each folders have been cleaned by removing or cropping images with a white background, people, frames, staged photography, or any other outliers that don't resemble photos taken in nature. This helps the app target images taken outside, for natural spottings of mushrooms. 

The base of the mushroom identification app is a [Tensorflow Keras](https://keras.io/) convolutional neural network model. The identification of the mushroom species is done in a two step process, by first classifying the taxonomical order of the mushroom and then the species of the mushroom. The information for the taxonomical lineage is webscraped from the [NCBI website](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi) using [Selenium's Webdriver](https://www.selenium.dev/documentation/webdriver/). The species of mushrooms are then labeled into 12 different orders.

The images from the folders are taken and read into [NumPy](https://numpy.org/doc/) arrays using [scikit-image](https://scikit-image.org/docs/stable/api/skimage.html) and split into training, testing, and validation lists using [scikit-learn](https://scikit-learn.org/stable/index.html), grouping the images both by order and by species. The orders of the mushroom species that have only one species are considered as identified species, and the rest of these orders become second order CNN models to then identify the images by species. The images are shuffled, resized into 112x112x3 arrays, and the images and target labels are converted into tensors, preparing them for the Keras machine learning model.

The convolutional neural network (CNN) model for identifying the taxonomical orders of the mushrooms, as well as the subsequent CNN models used to identify the mushroom species share the same basic features. The first is image augmentation using ImageDataGenerator which rotates, shifts and flips the images to improve accuracy. Second, the model includes three convoluional layers, each with batch normalization, max pooling and dropout layers to prevent overfitting. The model is then compiled using categorical cross-entropy as the loss function and the Adam optimizer. The accuracy of each model, using the test image data, is then saved for calculation of the global accuracy of the model, which is around 41%.

Data for each species of mushroom is downloaded from the [GlobalFungi Database](https://globalfungi.com/) in the form of Pandas dataframes. The datasets contain information about environmental variables for a multitude of mushroom samples for each species. These include latitude, longitude, continent, sample type, biome, mean annual temperature (MAT), mean annual precipitatio (MAP), and pH. These environmental features are collected to form a report about each identified species in the deployment of the model. Moreover, spottings of each species found on the [Project Noah website](https://www.projectnoah.org) are webscraped and offer additional information about different spotted mushroom specimens, such as geographical location, spotting date and image. The [OpenCage](https://opencagedata.com/) API is then used to get geographical coordinates from the city name and country for each spotting location.

## Project Architecture
The deployement of the model is done using a [Flask](https://flask.palletsprojects.com/en/2.1.x/) app, which reconstruct the saved CNN models by loading the saved model weights using Keras, and also loads data such as the species names and corresponding labels dictionaries, as well as the metadata from the samples datasets, and the edibility for each species. The app takes in the input of a mushroom images, transforms it into an array of the right dimensions to be used by the neural network models and identifies the species of the mushroom in the image. The app then uses that species to form a report using the data associated with that species using [Plotly](https://plotly.com/python-api-reference/index.html) to display the samples data in the form of pie charts, geomaps, and histograms. The data from the spottings for that species is also retrieved and the first ten spottings are displayed on the app. The app uses two HTML templates, the first to submit the image and the second to show the report.

## User Guide
To use the project, download the repository files and extract the contents from the zip file. Make sure you have [python](https://www.python.org/downloads/) installed, and the following modules installed: keras, tensorflow, scikit-image, matplotlib, seaborn, and pandas. If you don't, open up a command prompt and install them by using 'pip install <module name>' for each module. Then, click on the python file named "flask_app" and Ctrl + Click to follow the local host link "http://127.0.0.1:5000/" that appears in the command prompt window. A new page should open in your browser where you can submit an image that has been saved to your computer. Alternatively, you can pick an image from the sample_images folder. Another way to run the app is to copy the path to the folder that contains the flask app, open a command prompt window by pressing the "Windows" key and "R", typing "cd + <the copied path>" and then "python flask_app.py" to start running the app. If you need any more help, follow this [tutorial](https://drive.google.com/file/d/1gcrCm33sqFTKnoUnQOmTCIw7TGQsUdbC/view?usp=sharing).

## Technical Documentation
The code that forms the project is provided in the files:
- notebook
    - mushroom_project_main: Main code for image data transformation and webscraping
    - mushroom_project_by_order: CNN model to classify images by order
    - order_0_mushroom_project_by_species: CNN model used to classify images of order 0 by species (similarly for order 2,3,5,6,8,10)

The files used to run the app:
- mushroom_models
    - mushroom_model: Saved model weights in file "mushroom_model" and model history
    - mushroom_model_order_0: Saved model weights in file "mushroom_model_order_0" and model history (similarly for order 2,3,5,6,8,10)
- pickle_jar
    - edibility_dict: Contains species name and edibility
    - order_dict: Contains labels and order names
    - species_conversion_dict: Contains labels of species for each second order model
    - species_dict_inverse: Contains species label and species name
    - species_images_links: Contains image link for each species
    - species_metadata_dict: Contains sample datasets for each species
    - spotting_list: Contains all spotting data
- templates
    - first_page: HTML template for welcome page
    - results: HTML template for results page
- sample_images: images that can be used as input
