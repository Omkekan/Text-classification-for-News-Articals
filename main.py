import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Clean and preprocess the input text."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Load the dataset
data = pd.read_csv('/content/test.csv')

print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Combine title and description
data['article_text'] = data['Title'].fillna('') + ' ' + data['Description'].fillna('')

# Preprocess the text
data['processed_text'] = data['article_text'].apply(preprocess_text)

# Prepare features and labels
X = data['processed_text']
y = data['Class Index']

print("Unique classes:", y.unique())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the classification pipeline
classifier = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LinearSVC())
])

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Define category names
categories = ['World', 'Sports', 'Business', 'Sci/Tech']

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

def classify_news(headline):
    """Classify a news headline into one of the predefined categories."""
    processed_headline = preprocess_text(headline)
    prediction = classifier.predict([processed_headline])[0]
    return categories[prediction - 1]

# Interactive classification
while True:
    user_input = input("\nEnter a news headline (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    predicted_category = classify_news(user_input)
    print(f"Predicted category: {predicted_category}")

    # Debug information
    processed_input = preprocess_text(user_input)
    print(f"Processed input: {processed_input}")
    probabilities = classifier.decision_function([processed_input])
    print(f"Decision function values: {probabilities}")

print("Thanks for using the news classifier!")
