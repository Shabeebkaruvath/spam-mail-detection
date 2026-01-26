import numpy as np
import skfuzzy as fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. SETUP (Mock Training to initialize the model)
# In a real scenario, replace these with your actual dataset records.
train_emails = [
    "Get cheap loans now, click here for free money", 
    "Meeting scheduled for tomorrow at 10am",
    "Win a lottery prize today! Urgent action required",
    "The project report is attached for your review"
]
labels = [1, 0, 1, 0] # 1 for Spam, 0 for Ham

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_train = vectorizer.fit_transform(train_emails).toarray()

# Apply Fuzzy Membership to the training features as per the paper
# This softens the 'crisp' TF-IDF values before training
X_fuzzy_train = fuzz.sigmf(X_train, 0.5, 10)

model = MultinomialNB()
model.fit(X_fuzzy_train, labels)

# 2. THE INPUT INTERFACE
def check_spam_rate():
    print("\n--- Email Spam Rate Checker ---")
    user_input = input("paste full email content here:\n")
    
    # Preprocessing the user input
    input_vector = vectorizer.transform([user_input]).toarray()
    
    # Apply the same Fuzzy Membership transformation
    input_fuzzy = fuzz.sigmf(input_vector, 0.5, 10)
    
    # Get probability instead of a binary label
    # predict_proba returns [prob_of_ham, prob_of_spam]
    probability = model.predict_proba(input_fuzzy)[0][1]
    
    spam_percentage = probability * 100
    
    print("-" * 30)
    print(f"Spam Rate: {spam_percentage:.2f}%")
    
    if spam_percentage > 50:
        print("Result: This email is likely SPAM.")
    else:
        print("Result: This email is CLEAN.")

# Run the checker
check_spam_rate()