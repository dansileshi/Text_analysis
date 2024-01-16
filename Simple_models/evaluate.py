import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from omegaconf import OmegaConf


def evaluate(config):
    print("Evaluating...")
    test_inputs = joblib.load(config.features.test_features_save_path)
    test_df = pd.read_csv(config.data.test_csv_save_path)

    test_outputs = test_df["label"].values
    class_names = test_df["sentiment"].unique().tolist()

    model = joblib.load(config.train.model_save_path)

    # Make predictions
    predicted_test_outputs = model.predict(test_inputs)
    
    # Calculate evaluation metrics
    confusion_list = confusion_matrix(test_outputs, predicted_test_outputs)
    auc_score = roc_auc_score(test_outputs, predicted_test_outputs)
    accuracy = accuracy_score(test_outputs, predicted_test_outputs)

    # Store metrics in result_dict
    result_dict = {
        "confusion_matrix": confusion_list.tolist(),
        "auc_score": float(auc_score),  
        "accuracy": float(accuracy),  
    }
    print(result_dict)
    # Save the result_dict to a YAML file
    OmegaConf.save(result_dict, config.evaluate.results_save_path)

if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    evaluate(config)
