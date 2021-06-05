#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 03:59:18 2020

@author: oceane salmeron 
"""
import pandas as pd 
import numpy as np
from collections import Counter


def clean_data(data):
    
    # Replace ham and spam by 0 and 1 respectively
    data.Type[data.Type == 'ham'] = 0
    data.Type[data.Type == 'spam'] = 1
    
    # Cleaning Data
    
    # Delete all kind of punctuation
    data["Mail"] = data['Mail'].str.replace(r'[^\w\s]','')
    
    # Change all words to lower caps
    data["Mail"] = data['Mail'].str.lower()

    return data


def separate_train_test(data):
    
    # Shuffle data
    shuffle = data.sample(frac=1)
    
    # Define train set size - We decided to have 80% for the train set and 20% for the test set
    train_size = int(0.8 * len(data))
    
    # Divide the data in train and test set
    train_set = shuffle[:train_size]
    test_set = shuffle[train_size:]
    
    # Separate x_train, y_train and x_test, y_test
    # x_train = train_set.iloc[:,-1:]
    # y_train= train_set.iloc[:, :1]
    # x_test = test_set.iloc[:,-1:]
    # y_test= test_set.iloc[:, :1]
    
    return train_set, test_set



def make_Dictionary(train_set):

    # Create an empty list for all the words
    all_words = []   
          
    # Return a list of the words of the string    
    emails = train_set['Mail'].str.split()
    
    # Iterate all emails to add every words
    for mail in emails:
        for word in mail:
            word = word.split()
            all_words += word
            
    # Unordered collection with elements are stored as dictionary keys and their counts are stored as dictionary values
    dictionary = Counter(all_words)
    
    # Clean the dictionnary
    for item in list(dictionary):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
             del dictionary[item]
    
    # Only keep the most common 3000 words
    dictionary = dictionary.most_common(3000)

    return dictionary



def extract_features(train_set, dictionary):
    
    # Return a list of the words of the string
    emails = train_set['Mail'].str.split()
    
    # Create a matrix full of zeros of the length of the emails and the dictionary
    features_matrix = np.zeros((len(emails),3000))
    
    # Initialize docId to 0
    docID = 0;
    
    # Iterate all mails to create the features matrix
    for mail in emails:
      for word in mail:
          wordID = 0
          for i,d in enumerate(dictionary):
              if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = word.count(word)
      docID = docID + 1     
    
    # Concat the data set with the features matrix to have a clear visualition of the data
    data = pd.concat([train_set.reset_index(), pd.DataFrame(features_matrix)], axis=1).iloc[:,1:]

    return features_matrix, data



def compute_parameters(train_set, dic):
    
    # Separate all hams and spams
    spam = train_set[train_set['Type'] == 1]
    ham = train_set[train_set['Type'] == 0]
    
    # Laplace smoothing set to 1
    alpha = 1
    
    # Probability of ham and spam
    priorHam = len(ham)/len(train_set)
    priorSpam = len(spam)/len(train_set)
    
    # Total number of the words in ham mails
    nHam = (ham['Mail'].apply(len)).sum()
    # Total number of the words in spam mails
    nSpam = (spam['Mail'].apply(len)).sum()

    
    # Number of unique words in the dictionary
    nDic = len(dic)
        
    print('\n      > Compute parameters done \n')
    return alpha, priorHam, priorSpam, nHam, nSpam, nDic



def compute_pWSpam(data, word, nSpam, nDic, alpha):
    
    # Naive Bayes for P(Wi|Spam)
    return (data.loc[data['Type'] == 1, word].sum() + alpha) / (nSpam + alpha*nDic)



def compute_pWHam(data, word, nHam, nDic, alpha):
    
    # Naive Bayes for P(Wi|Ham)
    return (data.loc[data['Type'] == 0, word].sum() + alpha) / (nHam + alpha*nDic)



def classify(emails, priorHam, priorSpam, alpha, nHam, nSpam, nDic, dic, data):
    
    # Initialize P(Spam) and P(Ham)
    pSpamMail = priorSpam
    pHamMail = priorHam
    
    # Create a dataframe for the prediction
    df = pd.DataFrame(columns=['Prediction'])

    # Iterate all mails to classify
    for mail in emails.str.split():
        
        for word in mail:
            
            for i,d in enumerate(dic):
                # If the word is in the dictionnary we compute the probability, if not we dont
                if d[0]==word:
                    wordID = i
                    
                    pSpamMail *= compute_pWSpam(data, wordID, nSpam, nDic, alpha)
                    pHamMail *= compute_pWHam(data, wordID, nHam, nDic, alpha)
                
            
        if pHamMail > pSpamMail:
            
            df = df.append({'Prediction': 0},ignore_index=True)
            
        elif pHamMail <= pSpamMail:
            
            df = df.append({'Prediction': 1},ignore_index=True)

        pSpamMail = priorSpam
        pHamMail = priorHam
            
           
    # Concat the data we're classifying with the prediction
    data = pd.concat([data.reset_index(), df], axis=1).iloc[:,1:]
    return data



def classify_all(data, train_dic):
    
    # Compute all parameters to fit the model to the train set
    alpha, priorHam, priorSpam, nHam, nSpam, nDic = compute_parameters(train_set, train_dic)
    
    # Classify all mails and return it
    return classify(data['Mail'], priorHam, priorSpam, alpha, nHam, nSpam, nDic, train_dic, data)
    
    
    
def measure_performance(data):
    
    # Initialize the number of correct predictions to 0
    correct = 0
    
    # Initialize the number of True Positive, True Negative, False Positive, False Negative to 0
    TP = TN = FP = FN = 0
    
    # Instance two list to keep track of the true condition and the predicted condition
    true_list = []
    pred_list = []
    
    # Iterate every data rows
    for row in data.iterrows():
        
        row = row[1]
        
        # Append the result to the respective list
        true_list.append(row['Type'])
        pred_list.append(row['Prediction'])
        
        if row['Type'] == row['Prediction']:
            
            correct += 1
            
            if row['Type'] == 0:
                TP += 1
            elif row['Type'] == 1:
                TN += 1
        else:
            
            if row['Type'] == 0:
                FP += 1
            elif row['Type'] == 1:
                FN += 1
            
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    spec = TN / (TN + FP)
    f1 = 2*(precision*recall) / (precision + recall)
    
    true_condition = pd.Series(true_list, name='true conditions', dtype=np.intc)
    predicted_condition = pd.Series(pred_list, name='predicted conditions', dtype=np.intc)
    confusion_mx = pd.crosstab(predicted_condition, true_condition)
    
    print('Correct predictions:', correct)
    print('\nIncorrect predictions:', len(data) - correct)
    print('\nAccuracy:', acc)
    print('\nPrecision:', precision)
    print('\nRecall:', recall)
    print('\nSpecificity:', spec)
    print('\nF1 score:', f1)
    print('\nTP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
    print('\nConfusion matrix:\n\n', confusion_mx)

    
    
if __name__ == "__main__":   
    # Import data from txt file
    data = pd.read_csv('./data/messages.txt', sep="	", header=None, names=['Type', 'Mail'])

    # Clean data
    data = clean_data(data)
    print('\n> Cleaning Data done \n')


    print('\n> Data describtion by type (ham or spam) \n')
    print(data.groupby('Type').describe())

    # Divide the data in two groups train and test
    train_set, test_set = separate_train_test(data)
    print('\n> Separate train test done \n')

    # Print the shape of both train and test set
    print('\n> Train and test dataframe shapes \n')
    print('train set:', train_set.shape)
    print('\ntest set:', test_set.shape)

    # Parse the training and test set
    print('\n> Analysing train and test dataframe\n')
    print('train set:\n', train_set['Type'].value_counts(normalize=True))
    print('\ntest set:\n', test_set['Type'].value_counts(normalize=True))

    # Create dictionary
    train_dic = make_Dictionary(train_set)
    print('\n> Dictionary done \n')

    # Create features matrix for both train and test set in order to compute naive bayes 
    print('\n> Computing features matrix... \n')
    features_matrix_train, train_data_clean = extract_features(train_set, train_dic)
    features_matrix_test, test_data_clean = extract_features(test_set, train_dic)
    print('\n> Features matrix done \n')

    # Classify all the emails from a set
    print('\n> Classification... \n')
    test_data_clean = classify_all(test_data_clean, train_dic)
    print('\n> Classification done \n')

    # Compute model performance and display confusion matrix
    print('\n> Performance \n')
    measure_performance(test_data_clean)








