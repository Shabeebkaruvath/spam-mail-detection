import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and prepare dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Create balanced dataset splits
spam_emails = df[df['label'] == 1].reset_index(drop=True)
non_spam_emails = df[df['label'] == 0].reset_index(drop=True)

# Training: 600 non-spam + 300 spam
non_spam_training = non_spam_emails.iloc[:600]
spam_training = spam_emails.iloc[:300]
training_data = pd.concat([non_spam_training, spam_training], ignore_index=True)
training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Test: remaining emails
non_spam_test = non_spam_emails.iloc[600:800]
spam_test = spam_emails.iloc[300:400]
test_data = pd.concat([non_spam_test, spam_test], ignore_index=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing and term extraction
train_messages = training_data['message'].values
train_labels = training_data['label'].values
test_messages = test_data['message'].values
test_labels = test_data['label'].values

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_messages).toarray()
X_test = vectorizer.transform(test_messages).toarray()

# Fuzzy membership transformation
X_fuzzy_train = fuzz.sigmf(X_train, 0.3, 15)
X_fuzzy_test = fuzz.sigmf(X_test, 0.3, 15)

# Train Naïve Bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_fuzzy_train, train_labels)

# Evaluate model
test_predictions = model.predict(X_fuzzy_test)
accuracy = accuracy_score(test_labels, test_predictions)

print(f"Model Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, target_names=['Non-SPAM', 'SPAM']))

# Interactive email spam checker
def check_spam_rate():
    user_input = input("\nEnter email content (or 'quit' to exit):\n")
    
    if user_input.lower() == 'quit':
        return False
    
    if not user_input.strip():
        print("Error: No content provided.")
        return True
    
    # Transform and predict
    input_vector = vectorizer.transform([user_input]).toarray()
    input_fuzzy = fuzz.sigmf(input_vector, 0.3, 15)
    prediction = model.predict(input_fuzzy)[0]
    probability = model.predict_proba(input_fuzzy)[0]
    spam_percentage = probability[1] * 100
    
    print(f"\nSPAM: {spam_percentage:.2f}% | HAM: {probability[0]*100:.2f}%")
    
    if spam_percentage > 50:
        print(f"Result: SPAM (confidence: {spam_percentage:.1f}%)")
    else:
        print(f"Result: HAM (confidence: {probability[0]*100:.1f}%)")
    
    return True

# Main execution
if __name__ == "__main__":
    print("\nSpam Email Classification System Ready")
    check_spam_rate()