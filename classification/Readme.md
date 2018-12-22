# Road Sign Classification

This module includes the code associated with the training and testing of the classification model for classifying the detected road sign into 43 categories.
We used the generated model to predict the approching roadsign.
The model outputs the classID of th road sign in the gtrsb_kaggle.csv file. Then we use the sign_name.csv, which is mapping from classID to the name of the road sign, to get the name of the predicted road sign.

To use this model and get the name of predicted road sign use -
python3 evaluate4.py --model <path_to_model>

This will predict the name of all road signs in the images stores in the data folder under test_images.
