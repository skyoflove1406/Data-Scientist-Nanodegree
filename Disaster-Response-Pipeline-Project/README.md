# Disaster Response Pipeline Project

Udacity Data Scientist Nanodegree Project.


## Installation

Firstly, You will need to clone source from the repos
```bash
git clone https://github.com/skyoflove1406/Data-Scientist-Nanodegree.git
cd Disaster-Response-Pipeline-Project
```
This code runs with Python version 3.* and requires some libraries, to install theses libraries you will need to execute:
```bash
pip install -r requirements.txt
```


## Project Introduction

This is an Udacity Nanodegree project.

The project's purpose is building ETL data, Natural Language Processing and Machine Learning Model Pipeline to categorize tweet on the dataset which provided by Figure Eight Dataset
## File Description
app<br />
| - template<br />
| |- master.html # main page of web app<br />
| |- go.html # classification result page of web app<br />
|- run.py # Flask file that runs app<br />

data<br />
|- disaster_categories.csv # data to process<br />
|- disaster_messages.csv # data to process<br />
|- process_data.py<br />
|- InsertDatabaseName.db # database to save clean data to<br />
models<br />
|- train_classifier.py<br />
|- classifier.pkl # saved model<br />

README.md<br />
#### Executing Program
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
   ```bash
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
    - To run ML pipeline that trains classifier and saves
   ```bash
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

2. Go to `app` directory: 
```bash 
cd app 
```

3. Run your web app:
   ```bash
   python run.py
   ```
4. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements

Figure Eight for providing the dataset.