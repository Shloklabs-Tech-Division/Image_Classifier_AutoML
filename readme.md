## `This is an OpenSource Initiative to support the young talents led by SHLOKLABS & PSG College of Technology`

# Image Classifier AutoML

A flask web app written using python, which acts as an web interface to the teacheable machines project

## About the teacheable machines project(Image classification model selector):

This helps the users to reduce time in building their ML projects based on image classification.
The user simply has to provide the training image data in the required format.
The system will use the training data and will train 5 deep learning models.
The models' hyper-parameters will be tuned for the given training data.
The training data will be automatically balanced among the classes.
The trained 5 models will be used by the ModelSelector class to give the best prediction based on hard voting, thus increasing the accuracy, but this requires high amount of resources.

## About this project

Using flask, provides a web interface for the teacheable machines module, thus helps the users to easily interact with system.
The Image classification model selector is used as a package in this project.
To run this project just launch the index.py using the command `python index.py`, for this to run the system should have python installed.
Install any packages that is required to launch this app.

## required packages

1. pip (needed to install other packages)
2. tensorflow
3. shutil
4. flask
5. numpy
6. sklearn
7. PIL
8. keras
9. tensorflow_hub
10. keras_tuner
11. moviepy

## pages in the web app

first page to upload the images/video for the training data.
<img width="960" alt="ip4" src="https://user-images.githubusercontent.com/66177629/174311310-d566581b-eba5-40e5-902c-78f511826fca.png">

second page to configure the system for training.
<img width="960" alt="ip3" src="https://user-images.githubusercontent.com/66177629/174311284-9a565927-dab3-4101-94b5-8229d17da644.png">

third page to upload the image and initiate the prediction. The models can also be exported as a zip file.
<img width="923" alt="ip5" src="https://user-images.githubusercontent.com/66177629/174311403-650bf175-d421-406f-a56e-912388b00ba0.png">
<img width="960" alt="ip1" src="https://user-images.githubusercontent.com/66177629/174311244-8d1eb307-2812-43d8-938f-8c3cbc9fb6a1.png">

---

To learn about the **Image classification model selector** project refer : [git link to Image classification model selector](https://github.com/Krishna-Teja732/Img_classification_model_selector)

---

Thanks for SHLOKLABS to provide us with the opportunity to work in the project.
 
A special Thanks **Manoj Kumar Thangadurai**, _Lead Engineer, Shloklabs_ and **Dr. N. Arulanand**, _Professor,PSG College of Technology_ for mentoring us throughtout the project and giving us their valuable feedback.


## Contributions

### Mentors:
1. `Dr. N. Arulanand, Professor, PSG College of Technology`
2. `Muralitharan Sivalingam, CTO, SHLOKLABS`
3. `Manoj Kumar Thangadurai, Lead Engineer, Shloklabs`

### Core team:
1. Gautham S
2. Kamalraj D
3. Krishna Teja B
4. Pawan Kumar
5. Dhanush Reddy
6. Srikanth

![shloklabs-squarelogo](https://shloklabs.com/wp-content/uploads/2021/01/shlok-300x30.png)

![PSG College Logo](https://1.bp.blogspot.com/-l_CBUtQxwRk/XmkkGnEuVZI/AAAAAAAA0zU/FpQtqOiERbgg9RiaHLCegQEY-UuLzlkZgCLcBGAsYHQ/s1600/images%2B%25282%2529.png)
