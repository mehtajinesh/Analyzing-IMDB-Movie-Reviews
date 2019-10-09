#for reading files
import os
import glob

#for shuffling postive and negative reviews
from sklearn.utils import shuffle

#for removing stop words and stemming the words in review
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup

#caching the preprocessed reviews
import pickle

#storing preprocess words in numpy array
import numpy as np

#store the processed data into train.csv using dataframe
import pandas as pd

#training uploading train data to s3
import sagemaker
import boto3

#for loading env variable
from os.path import join, dirname
from dotenv import load_dotenv

#training model using pytorch
import torch
import torch.utils.data

import torch.optim as optim
#for import train module
import sys
sys.path.insert(0, 'train/')
from model import LSTMClassifier
from train import train
from sagemaker.pytorch import PyTorch
from sklearn.metrics import accuracy_score
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel

def read_imdb_data(data_dir='data\\aclImdb'):
    """Reading imdb data from the files"""
    
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            for f in files:
                with open(f,encoding="utf8") as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test

def review_to_words(review):
    """Converting the review to meaningful words by stemming unneccessary information."""
    
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [stemmer.stem(w) for w in words] # stem
    
    return words

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        print('Working on it.')
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        print('Training Done.')
        words_test = [review_to_words(review) for review in data_test]
        print('Testing Done.')        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.

    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    
    for review in data:
        for word in review:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    # Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    
    sorted_words = [item[0] for item in sorted(word_count.items(), key=lambda x: x[1], reverse=True)]
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict

def convert_and_pad(word_dict, sentence, pad=500):
    """Mapping words to integers and adding extra 0 if the review size < 500"""
    
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)

def loadEnv():
    """ 
    Loads the enviroment vars like access key info
    to the file 
    """
    # Create .env file path.
    dotenv_path = join(dirname(__file__), '.env')
    # Load file from the path.
    load_dotenv(dotenv_path)
    return

def predict(pytorch_predictor, data, rows=512):
    # We split the data into chunks and send each chunk seperately, accumulating the results.
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, pytorch_predictor.predict(array))
    return predictions

def main():
    data, labels = read_imdb_data()
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    #storing the preprocess data as cache
    cache_dir = os.path.join("cache", "sentiment_analysis")  # where to store cache files
    os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists
    # Preprocess data
    train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y,cache_dir)
    #building word dict from reviews
    word_dict = build_dict(train_X)
    #now we store word dict for future references
    data_dir = 'data/pytorch' # The folder we will use for storing data
    if not os.path.exists(data_dir): # Make sure that the folder exists
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
        pickle.dump(word_dict, f)

    train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
    test_X, test_X_len = convert_and_pad_data(word_dict, test_X)

    #store processed data
    pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
    
    loadEnv()
    # Accessing variables.
    access_key_id = os.getenv('ACCESS_KEY_ID')
    secret_key = os.getenv('SECRET_KEY')
    region = os.getenv('AWS_REGION')
    execution_role = os.getenv('EXEC_ROLE')
    # create sagemaker session
    session = boto3.Session(aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_key,region_name=region)

    sagemaker_session = sagemaker.Session(boto_session=session)

    #update data to s3 bucket
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/sentiment_rnn'
    role = execution_role
    input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
    
    # Read in only the first 250 rows
    train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=250)

    # Turn the input pandas dataframe into tensors
    train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
    train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

    # Build the dataset
    train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
    # Build the dataloader
    train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMClassifier(32, 100, 5000).to(device)
    optimizer = optim.Adam(lstm_model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(lstm_model, train_sample_dl, 5, optimizer, loss_fn, device)

    estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.m4.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })
    estimator.fit({'training': input_data})

    # Deploy the trained model
    class StringPredictor(RealTimePredictor):
        def __init__(self, endpoint_name, sagemaker_session):
            super(StringPredictor, self).__init__(endpoint_name, sagemaker_session, content_type='text/plain')

    py_model = PyTorchModel(model_data=estimator.model_data,
                        role = role,
                        framework_version='0.4.0',
                        entry_point='predict.py',
                        source_dir='serve',
                        predictor_cls=StringPredictor)
    pytorch_predictor = py_model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
    print(pytorch_predictor.endpoint)
    return
main()