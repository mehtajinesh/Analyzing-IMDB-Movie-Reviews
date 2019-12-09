# Sentimental-Analysis-using-PyTorch

This project allows a user to provide their review regarding a movie and using Sentimental Analysis, the underlying model predicts whether the review is positive or negative. The training model is trained using Pytorch NN and training data used is IMDB review dataset available at https://www.kaggle.com/utathya/imdb-review-dataset.

## Getting Started

### Clone

- Clone this repo to your local machine by typing the following command :

  ```
  $ git clone `https://github.com/mehtajinesh/Sentimental-Analysis-using-NN.git`
  ```

### Prerequisites

Things to install before starting :

- Python 3.7 'https://www.python.org/downloads/release/python-370/'

- AWS CLI 

  ```pip install awscli```
  
  - once the installation is done, configure the system with your aws_key_id, aws_key and region by doing the following:
  
    ```aws configure```

## Deployment Steps

1. First step is to install all the modules required to run the python code into your local system. Use the following command:

  ```
  $ pip install -r requirements.txt
  ```
  
2. Once all the modules are installed, then download the IMDB dataset and extract folders for training the model.
  ```
  %mkdir ../data
  !wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  !tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
  ```
3. Next, we combine the training and deployment (elaborated below) inside the main.py file :
  - preprocessing data
  - training model by fitting the data to pytorch model
  - deploy the trained model on sagemaker which will constantly interact with the AWS Lambda function
  
    So for training the model, we execute the main.py file like this :
  ```
  $python main.py
  ```
4. Once the execution is done, it will print out the deployed model name, which is provided to the AWS Lambda function.

5. Next, we create a AWS API Gateway which will communicate to the AWS Lambda function whenever a submit request comes.

6. Once the API is ready, we update the URL inside the index.html file and voila! We are good to go!!

## Built With

* [Python](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [AWS SageMaker](https://aws.amazon.com/sagemaker/) - where the trained model are deployed
* [AWS Lambda](https://aws.amazon.com/lambda/) - where the prediction is done using trained model
* [AWS S3](https://aws.amazon.com/s3/) - where the model artifacts are stored
* [AWS API Gateway](https://aws.amazon.com/api-gateway/) - where the url hits on click of submit

## Authors

* **Jinesh Mehta** 

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2019 © <a href="http://jineshmehta.com" target="_blank">Jinesh Mehta</a>.

### **Reviews & Validations**
- From [Udacity](https://github.com/mehtajinesh/Sentimental-Analysis-using-PyTorch/blob/master/review.pdf)
