import pandas as pd
from transformers import pipeline

# load dataset

def load_dataset(file_path):
    """
    load dataset from a csv file.
    args:
        file_path (str): path to the csv file containing the dataset.
    returns:
        pd.dataframe: loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print("dataset loaded successfully!")
        print(f"first 5 rows:\n{data.head()}")
        return data
    except Exception as e:
        print(f"error loading dataset: {e}")
        return None

# pre-trained sentiment analysis model

def load_pretrained_model():
    """
    load a pre-trained distilbert model for sentiment analysis.
    returns:
        pipeline: a hugging face pipeline for text classification.
    """
    try:
        model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        print("model loaded successfully!")
        return model
    except Exception as e:
        print(f"error loading model: {e}")
        return None

# predict sentiment

def predict_sentiment(model, texts):
    """
    predict sentiment for a list of texts using the pre-trained model.
    args:
        model (pipeline): hugging face pipeline for text classification.
        texts (list): list of texts to analyze.
    returns:
        list: list of predictions with labels and confidence scores.
    """
    try:
        predictions = model(texts)
        print("\npredictions:")
        for text, pred in zip(texts, predictions):
            print(f"text: {text}\npredicted: {pred['label']}, confidence: {pred['score']:.2f}\n")
        return predictions
    except Exception as e:
        print(f"error during prediction: {e}")
        return []

# save predictions

def save_predictions(predictions, output_file):
    """
    save predictions to a csv file.
    args:
        predictions (list): list of prediction results.
        output_file (str): path to the output csv file.
    """
    try:
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        print(f"predictions saved to {output_file}")
    except Exception as e:
        print(f"error saving predictions: {e}")

# main execution block

if __name__ == "__main__":
    
    # define file paths

    dataset_path = "mental_health_dataset.csv"
    output_path = "predictions.csv"

    # load dataset

    dataset = load_dataset(dataset_path)
    if dataset is None:
        exit()

    # extract sample texts

    sample_texts = [
        "i'm feeling great and hopeful about the future.",
        "i feel so sad and overwhelmed by everything.",
        "life is just okay, nothing special.",
        "social situations make me feel anxious."
    ]

    # load pre-trained model

    model = load_pretrained_model()
    if model is None:
        exit()

    # predict sentiment

    predictions = predict_sentiment(model, sample_texts)
    
    # adjust predictions for specific cases

def adjust_predictions(text, prediction):
    """
    adjust predictions for ambiguous or neutral cases based on specific keywords.
    args:
        text (str): the input text.
        prediction (dict): the original prediction.
    returns:
        dict: adjusted prediction with label and score.
    """
    if "okay" in text.lower() or "nothing special" in text.lower():
        return {"label": "neutral", "score": 1.0}  # Corrected line
    return prediction

# apply adjustments to predictions
adjusted_predictions = [adjust_predictions(text, pred) for text, pred in zip(sample_texts, predictions)]

# display adjusted predictions
print("\nadjusted predictions:")
for text, pred in zip(sample_texts, adjusted_predictions):
    print(f"text: {text}\nadjusted predicted: {pred['label']}, confidence: {pred['score']:.2f}\n")



    # save predictions to csv

    save_predictions(predictions, output_path)
    
    # save adjusted predictions to a csv file
    
adjusted_predictions_df = pd.DataFrame(
    [{"text": text, "adjusted_label": pred["label"], "confidence": pred["score"]} 
     for text, pred in zip(sample_texts, adjusted_predictions)]
)
adjusted_predictions_df.to_csv("adjusted_predictions.csv", index=False)
print("Adjusted predictions saved to adjusted_predictions.csv")

    
