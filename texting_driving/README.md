
# Texting and Driving Project 

## Overview 
- The goal of this project was to see if we could detect texting from 
a video stream. 

## Training the Model on AWS 

- Launch instance from N. California 
- Use 'ami-125b2c72' on a g2.2xlarge instance 
- Log-in into the instance 
- Enter $vi ~/.theanorc
- Copy and paste the text from ./data/make_theano_work.txt 
- $pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
- $pip install https://github.com/Lasagne/Lasagne/archive/master.zip
- $pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
- $pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
- To make sure everything installed correctly, run $python -c "import theano; print theano.sandbox.cuda.dnn.dnn_available()"
- Make sure you see an output of True
- Clone this repository and cd into texting_driving directory 
- In the code directory place the texting_driving.MOV file
- Run $make data 
- Run $make run_model  
- Note: For more detailed instructions see http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/
