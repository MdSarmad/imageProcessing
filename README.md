# QUALITY CHECKING USING IMAGE PROCESSING

This is a mini project on quality of a fruit checking using image processing and deployed on production using Flask API

# PREREQUISITES

This project is using CNN algorithm(using keras) and Flask for API

# PROJECT STRUCTURE

This project has three major parts:
1. cnn.py : For building the model for image processing
2. app.py : This contains Flask APIs that takes an image as an input adn computes the result and return it.
3. templates : This folder contains the HTML template to allow user to insert image and diplay the desired result.

# RUNNING THE PROJECT

1. Ensure that you are in the project home directory and run the below command.

*python cnn.py

This would create model.h5

2. Run app.py using below command to start Flask API

*python app.py

By default, Flask will run on port 5000

3. Navigate to URL http://localhost5000

Now

Insert an image of a fruit to check the quality(Apple,Banana and Orange is used during model building).

And now you have the result.
