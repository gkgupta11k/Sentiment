# Import necessary libraries
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# Load the pretrained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Set up the sentiment analysis pipeline
nlp_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    return nlp_sentiment(text)

if __name__ == "__main__":
    # User input for analysis
    user_input = input("Enter the text for sentiment analysis: ")
    sentiment_result = analyze_sentiment(user_input)
    print(f"Sentiment: {sentiment_result[0]['label']} (Confidence: {sentiment_result[0]['score']:.2f})")
