# Sentiment Analysis Experiment

## Overview

This project performs sentiment analysis on the IMDb dataset. It includes scripts for preparing the data, training models, and evaluating the results.

## Getting Started

Follow these steps to run the experiment:

1. **Download the IMDb Dataset:**

   - Download the IMDb dataset from [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
   - Create a folder named `imdb_dataset` in the project directory.
   - Place the downloaded `imdb-dataset.csv` file inside the `imdb_dataset` folder.

2. **Configure Model and Parameters:**

   - Open the `params.yaml` file in the project directory.
   - You can change the type of model and its parameters in the configuration file. Modify the `model_type` and corresponding parameters as needed.

3. **Prepare and Train Data:**

   - Open a terminal or command prompt.
   - Navigate to the project directory.
   - Run the following command to prepare and train the data:

     ```bash
     python Prepare_Train_data.py
     ```

4. **Evaluate Results:**

   - After training is complete, run the evaluation script with the following command:

     ```bash
     python evaluate.py
     ```

5. **View Results:**

   - All results, including models and evaluation metrics, will be saved in a folder named `archive`.
