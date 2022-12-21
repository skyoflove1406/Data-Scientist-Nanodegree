# Data Scientist Capstone

Udacity Data Scientist Nanodegree Project.


## Installation

This code runs with Python version 3.* and requires some libraries, to install theses libraries you will need to execute:
```bash
pip install -r requirements.txt
```

You will need to download Starbuck's Capstone Challenge and put in data folders.


## Project Motivation

This is an Udacity Nanodegree project.

1. Problem Statement
In traditional marketing approach, the marketer does not know what is the offer the customer need?. 
Therefore, not only the business is not optimized but also get complained from customer. 
In this project, Customer profile, offer porfolio, and transaction data will be used to build predictive model 
for resolving the problem by predicting the chance the customer make the offer completed.

2. Project Goal
The project's purpose is data mining, data analysis, data transformation, and predicting the likelihood of customers using an offer if it is sent to a customer. Finally, improve the accuracy of the model.


## File Description

#### Starbucks_Capstone_notebook.ipynb: 
Notebook containing the data analysis/modelling.
#### data/portfolio.json: 
Offers sent during 30-day test period (10 offers x 6 fields)
    reward: (numeric) money awarded for the amount spent
    channels: (list) web, email, mobile, social
    difficulty: (numeric) money required to be spent to receive reward
    duration: (numeric) time for offer to be open, in days
    offer_type: (string) bogo, discount, informational
    id: (string/hash)

#### data/profile.json: 
Rewards program users (17000 users x 5 fields)
    gender: (categorical) M, F, O, or null
    age: (numeric) missing value encoded as 118
    id: (string/hash)
    became_member_on: (date) format YYYYMMDD
    income: (numeric)
#### data/transcript.json: 
Event log (306648 events x 4 fields)
    person: (string/hash)
    event: (string) offer received, offer viewed, transaction, offer completed
    value: (dictionary) different values depending on event type
    offer id: (string/hash) not associated with any "transaction"
    amount: (numeric) money spent in "transaction"
    reward: (numeric) money gained from "offer completed"
    time: (numeric) hours after start of test

## Results

The main findings of the explaination can be found at the post available [here](https://medium.com/@stephentran_45501/starbucks-capstone-challenge-new-way-for-sending-offer-to-customer-2298438a27d6)

## Licensing, Authors, Acknowledgements

Must give credit to Starbuck's Capstone Challenge for the data. You can find the Licensing for the data and other descriptive information at the Starbuck's Capstone Challenge link available [here](https://learn.udacity.com/nanodegrees/nd025/parts/cd1971/lessons/c20e1b63-c711-475b-b1ba-3ea987081193/concepts/547895d4-671c-4165-ae36-2938a4973f09).