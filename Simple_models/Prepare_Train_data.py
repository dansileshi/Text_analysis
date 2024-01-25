import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from omegaconf import OmegaConf

def prepare_data(config, dataset_path):
    """ Prepares the dataset """
    print("Preparing data...")
    df = pd.read_csv(dataset_path)
    df["label"] = pd.factorize(df["sentiment"])[0]
    print("sample data: ")
    print(df.head(3))  # Look at the first 3 rows of the dataframe

    test_size = config.data.test_set_ratio
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["sentiment"], random_state=321)

    train_df.to_csv(config.data.train_csv_save_path, index=False)
    test_df.to_csv(config.data.test_csv_save_path, index=False)

def make_features(config, model_type):
    """ Creates features from the text data """
    print("extracting features...")
    train_df = pd.read_csv(config.data.train_csv_save_path)
    test_df = pd.read_csv(config.data.test_csv_save_path)

    vectorizer_name = config.features.vectorizer
    vectorizer = {
        "tfidf-vectorizer": TfidfVectorizer
    }[vectorizer_name](stop_words="english")

    train_inputs = vectorizer.fit_transform(train_df["review"])
    test_inputs = vectorizer.transform(test_df["review"])

    joblib.dump(train_inputs, config.features.train_features_save_path)
    joblib.dump(test_inputs, config.features.test_features_save_path)

def train(config):
    print("Training...")

    # Load train inputs and outputs
    train_inputs = joblib.load(config.features.train_features_save_path)
    train_outputs = pd.read_csv(config.data.train_csv_save_path)["label"].values

    model_type = config.train.model_type.lower()
    # chagne the model type in the  config file
    if model_type == 'logistic_regression':
        model = LogisticRegression(**config.train.logistic_regression_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**config.train.random_forest_params)
    elif model_type == 'neural_network':
        model = MLPClassifier(**config.train.neural_network_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Print information about the model and its parameters
    print(f"Using model type: {model_type}")
    if model_type == 'logistic_regression':
        print(f"Model parameters: {config.train.logistic_regression_params}")
    elif model_type == 'random_forest':
        print(f"Model parameters: {config.train.random_forest_params}")
    elif model_type == 'neural_network':
        print(f"Model parameters: {config.train.neural_network_params}")

    # Train the model
    model.fit(train_inputs, train_outputs)

    # Save the trained model to a file
    joblib.dump(model, config.train.model_save_path)

def main(config):

    # Create the "archive" folder if it doesn't exist
    archive_folder = './archive'
    if not os.path.exists(archive_folder):
        os.makedirs(archive_folder)



    # Modify the path to the CSV file
    dataset_csv_path = './imdb_data/IMDB Dataset.csv' 

    # Ensure the dataset is present
    if not os.path.exists(dataset_csv_path):
        raise FileNotFoundError(f"File not found: {dataset_csv_path}")

    # Prepare data
    prepare_data(config, dataset_csv_path)

    # Create features based on model_type
    make_features(config, config.train.model_type)

    # Train the model
    train(config)


if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    main(config)
