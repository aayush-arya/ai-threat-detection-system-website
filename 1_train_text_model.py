import pandas as pd  # <--- THIS IS THE MISSING LINE!
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# --- 1. Load Real Dataset (SMS Spam Collection) ---
FILE_PATH = 'spam.csv' 

try:
    # ⚠️ Load with explicit column names, ignoring the extra empty columns
    # We use names=['label', 'text'] and then ignore the rest with `usecols`
    df = pd.read_csv(
        FILE_PATH, 
        encoding='latin-1', 
        header=None,
        sep=',',  # CRITICAL: Sometimes the SMS dataset is tab-separated, not comma-separated!
        names=['label', 'text', 'col3', 'col4', 'col5'],
        usecols=['label', 'text'] # ONLY read the first two columns
    )
except FileNotFoundError:
    print(f"ERROR: Dataset file '{FILE_PATH}' not found. Please check your file path and name.")
    exit()

# --- 2. Data Cleaning & Label Mapping ---
# 2a. Remove any remaining rows where the text or label is missing
df.dropna(inplace=True)

# 2b. Explicitly ensure the text column is a string (prevents NaN from sneaking in)
df['text'] = df['text'].astype(str)

# 2c. Drop any exact duplicate rows
df.drop_duplicates(inplace=True)

# ... (Lines 1-35: Data Loading and Initial Cleaning) ...

# ... (Lines 1-35: Data Loading and Initial Cleaning) ...

# 2d. Map the dataset's labels ('ham'/'spam') to your project's labels ('Safe'/'Threat')
df['label'] = df['label'].map({'ham': 'Safe', 'spam': 'Threat'})

print(f"Loaded {len(df)} unique samples after cleaning.")
print(f"Threat Count: {df[df['label'] == 'Threat'].shape[0]} / Safe Count: {df[df['label'] == 'Safe'].shape[0]}")

# --- 3. Text Preprocessing & Feature Extraction (TF-IDF) ---

# ⚠️ AGGRESSIVE FINAL CLEANING AND PREPARATION: 

# 3a. Force text to string, handle potential NaNs again, and reset index
df['text'] = df['text'].astype(str).fillna('') 
df['label'] = df['label'].astype(str).fillna('Safe') # Ensure label is also a string and non-NaN
df = df[df['text'].str.strip() != ''] # Remove empty messages
df.reset_index(drop=True, inplace=True) # Reset index to prevent alignment issues in split

# 3b. Separate X and y variables (using .values ensures a clean NumPy array)
X_data = df['text'].values
y_data = df['label'].values


print(f"Final sample count for training: {len(X_data)}")

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Apply vectorizer to the clean data array
X = vectorizer.fit_transform(X_data)
y = y_data # Use the clean y_data array

# 4. Model Training
# Split data into training (80%) and testing (20%)
# Use X and y that were created from the clean data arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Support Vector Classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train) 
# ... (Rest of the script for evaluation and saving)
# 5. Evaluation and Saving
predictions = model.predict(X_test)
print("\n--- Model Training Report on Real Data ---")
# This report shows Precision, Recall, and F1-score for your model
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model and the vectorizer for use in the Flask API
joblib.dump(model, 'text_threat_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\nSuccessfully trained and saved 'text_threat_model.pkl' and 'tfidf_vectorizer.pkl'.")