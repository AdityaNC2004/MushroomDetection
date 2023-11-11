# Mushroom Identification & Report App


## Introduction
With the ability to identify up to 100 distinct mushroom species, this app offers valuable insights to users who encounter mushrooms in the wild. Whether you're curious about the edibility of a mushroom, interested in its habitat and environmental preferences, or simply want to learn more about the species' distribution, this mushroom identification tool has you covered. It's a handy resource for mushroom enthusiasts and nature lovers seeking to deepen their understanding of the fungal world.

## Project Overview
This project leverages a Kaggle dataset [kaggle dataset](https://www.kaggle.com/datasets/derekkunowilliams/mushrooms)) containing images of mushrooms, organized into folders by species and categorized by edibility, ranging from deadly and poisonous to conditionally edible or completely safe to eat. The Python code for this initiative utilizes the OS library [OS](https://docs.python.org/3/library/os.html) library in conjunction with [Pandas](https://pandas.pydata.org/) to streamline the selection of 100's mushroom species with the most extensive image data. This selection aims for an equal distribution between edible and non-edible species. To enrich the chosen folders, additional mushroom images for each species are sourced from the internet and saved into the corresponding directories using a [zip file extension](https://download-all-images.mobilefirst.me/). The images within each folder undergo a cleaning process that involves the removal or cropping of pictures with white backgrounds, people, frames, staged photography, or other outliers that deviate from natural outdoor mushroom photographs. This meticulous curation ensures that the app primarily handles images taken in natural environments, reflecting real mushroom sightings.

The core of the mushroom identification app relies on a convolutional neural network (CNN) model implemented using TensorFlow's Keras[Tensorflow Keras](https://keras.io/). The identification process occurs in two steps: first, by classifying the taxonomical order of the mushroom, and then, by identifying the specific species. Information regarding the taxonomical lineage is obtained through web scraping from the [NCBI website](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi) using [Selenium's Webdriver](https://www.selenium.dev/documentation/webdriver/).The identified mushroom species are then grouped into 12 distinct orders.

The images from the folders are processed and transformed into NumPy arrays using scikit-image. Subsequently, the data is divided into training, testing, and validation sets using scikit-learn, taking into account the grouping of images both by order and by species. Orders of mushroom species containing only one species are considered as identified, while the remaining orders serve as second-order CNN models to pinpoint species. The images undergo shuffling, resizing into 112x112x3 arrays, and conversion into tensors to prepare them for use in the Keras machine learning model.

The CNN model architecture employed for identifying the taxonomical orders and subsequent CNN models for species identification share common features. Firstly, image augmentation using ImageDataGenerator is applied, which introduces variations in the images through rotation, shifting, and flipping to enhance accuracy. Secondly, each model consists of three convolutional layers, each accompanied by batch normalization, max-pooling, and dropout layers to mitigate overfitting. The models are then compiled using categorical cross-entropy as the loss function and the Adam optimizer. The accuracy of each model is evaluated using the test image data, and these individual accuracies are used to calculate the global accuracy of the overall model, which stands at approximately 41%.

Data pertaining to each mushroom species is downloaded from the[GlobalFungi Database](https://globalfungi.com/) in the form of Pandas dataframes. These datasets provide insights into environmental variables associated with numerous mushroom samples for each species. This information includes latitude, longitude, continent, sample type, biome, mean annual temperature (MAT), mean annual precipitation (MAP), and pH levels. These environmental factors serve as the basis for generating comprehensive reports about each identified species during the deployment of the model. Additionally, data regarding sightings of each species, as documented on the [Project Noah website](https://www.projectnoah.org), is collected through web scraping. This data includes geographical coordinates, spotting dates, and images of various mushroom specimens. The [OpenCage](https://opencagedata.com/) is then utilized to obtain geographical coordinates, translating city names and countries into specific locations for each spotting.

## Project Architecture
The deployement of the model is done using a [Flask](https://flask.palletsprojects.com/en/2.1.x/) app, which reconstruct the saved CNN models by loading the saved model weights using Keras, and also loads data such as the species names and corresponding labels dictionaries, as well as the metadata from the samples datasets, and the edibility for each species. The app takes in the input of a mushroom images, transforms it into an array of the right dimensions to be used by the neural network models and identifies the species of the mushroom in the image. The app then uses that species to form a report using the data associated with that species using [Plotly](https://plotly.com/python-api-reference/index.html) to display the samples data in the form of pie charts, geomaps, and histograms. The data from the spottings for that species is also retrieved and the first ten spottings are displayed on the app. The app uses two HTML templates, the first to submit the image and the second to show the report.

## User Guide
To use the project, download the repository files and extract the contents from the zip file. Make sure you have [python](https://www.python.org/downloads/) installed, and the following modules installed: "keras, tensorflow, scikit-image, matplotlib, seaborn, and pandas". If you don't, open up a command prompt and install them by using 'pip install <module name>' for each module.After downloading all modules run "flask_app" and run it and Ctrl + Click to follow the local host link "http://127.0.0.1:5000/" that appears in the terminal window. A new page should open in your browser where you can submit an image that has been saved to your computer. Alternatively, you can pick an image from the sample_images folder. Another way to run the app is to copy the path to the folder that contains the flask app, open a command prompt window by pressing the "Windows" key and "R", typing "cd + <the copied path>" and then "python flask_app.py" to start running the app.

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
