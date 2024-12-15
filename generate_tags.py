import pandas as pd
import py_vncorenlp
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Vietnamese Stop Words
stop_words = set()
try:
    with open('vietnamese-stopwords.txt', encoding='utf8') as f:
        stop_words = {line.strip() for line in f}
except FileNotFoundError:
    print("Stopwords file not found!")
    exit()

# 2. Load Vietnamese Dictionary
vietnamese_dict = set()
try:
    with open('tudien.txt', encoding='utf8') as f:
        vietnamese_dict = {line.strip() for line in f}
except FileNotFoundError:
    print("Vietnamese dictionary file not found!")
    exit()

# 3. Load Input CSV File
input_file = "articles_testing.csv"  # Replace with your CSV file path
try:
    df = pd.read_csv(input_file, encoding='utf8')
    if 'content' not in df.columns:
        raise KeyError("CSV file must contain a 'content' column.")
except FileNotFoundError:
    print("CSV file not found!")
    exit()
except KeyError as e:
    print(e)
    exit()


# 4. Stopword Removal Function
def removeStopWords(o_sen):
    return " ".join([word for word in o_sen.split() if word not in stop_words])


# 5. Download and Load VnCoreNLP
py_vncorenlp.download_model(save_dir=os.path.abspath('./vncorenlp'))
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath('./vncorenlp'))

# 6. Load Sentence Transformer
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# 7. Process Each Record
keywords_per_record = []  # To store keywords for each row
for index, row in df.iterrows():
    content = row['content']

    # Skip if content is NaN
    if pd.isna(content):
        keywords_per_record.append([])
        continue

    # Segment and clean content
    segmented_text = rdrsegmenter.word_segment(content)  # Segment text
    cleaned_text = " ".join([removeStopWords(sentence) for sentence in segmented_text])

    # Generate candidate words/phrases
    count = CountVectorizer(ngram_range=(1, 1))  # Adjust n-gram range for phrases
    count_fit = count.fit([cleaned_text])
    candidates = count_fit.get_feature_names_out()

    # Filter out candidates that are numbers or short words (length <= 4)
    candidates = [word for word in candidates if not word.isdigit() and len(word) > 4]

    # Compute embeddings
    if len(candidates) == 0:  # Check if there are any candidates left after filtering
        keywords_per_record.append([])
        continue

    doc_embedding = model.encode([cleaned_text])
    candidate_embeddings = model.encode(candidates)

    # Compute similarity and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    sorted_indices = distances.argsort()[0]  # Sort indices based on similarity

    # Extract keywords sorted by similarity
    keywords = [candidates[index] for index in sorted_indices]

    # Remove keywords that are in stop words
    filtered_keywords = [word for word in keywords if word not in stop_words]

    # Remove single words that exist in the Vietnamese dictionary
    final_keywords = [
        word for word in filtered_keywords if " " in word or word not in vietnamese_dict
    ]

    # Append the final keywords to the result
    keywords_per_record.append(final_keywords)

# 8. Add Keywords to DataFrame
df['tags'] = keywords_per_record

# 9. Save to CSV or Print
output_file = "nhom6_sol1.csv"
df.to_csv(output_file, index=False, encoding='utf8')

print(f"file saved as: {output_file}")
