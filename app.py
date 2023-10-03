import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your dataset with simulated ratings
df = pd.read_csv('C:/Users/Admin/Desktop/Hackthon/Bakery_with_SimulatedRating.csv')

# Label encoding for categorical variables
label_encoder = LabelEncoder()
df['DateTime'] = label_encoder.fit_transform(df['DateTime'])
df['Daypart'] = label_encoder.fit_transform(df['Daypart'])
df['DayType'] = label_encoder.fit_transform(df['DayType'])

# Collaborative Filtering (using Surprise's SVD)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['TransactionNo', 'Items', 'SimulatedRating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

cf_model = SVD(n_factors=100, n_epochs=20)  # Adjust parameters for SVD
cf_model.fit(trainset)
cf_predictions = cf_model.test(testset)

# Calculate MAE and RMSE for Collaborative Filtering
cf_true_ratings = [testset[i][2] for i in range(len(testset))]
cf_pred_ratings = [cf_predictions[i].est for i in range(len(cf_predictions))]

cf_mae = mean_absolute_error(cf_true_ratings, cf_pred_ratings)
cf_rmse = np.sqrt(mean_squared_error(cf_true_ratings, cf_pred_ratings))

# Content-Based Filtering (using scikit-learn)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Items'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to get content-based recommendations for a given item
def get_content_based_recommendations(item_name):
    item_index = df[df['Items'] == item_name].index[0]
    sim_scores = list(enumerate(cosine_sim[item_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar items (excluding itself)
    recommended_indices = [i[0] for i in sim_scores]
    recommended_items = df['Items'].iloc[recommended_indices].tolist()
    return recommended_items


# Hybrid Recommendation System with precision and recall calculation
def hybrid_recommendation(transaction_no):
    if transaction_no not in df['TransactionNo'].values:
        return "Transaction not found."

    # Find the items associated with the transaction
    purchased_items = df[df['TransactionNo'] == transaction_no]['Items'].tolist()

    # Get collaborative filtering predictions for each purchased item
    cf_predictions = {}
    for item in purchased_items:
        cf_predictions[item] = cf_model.predict(transaction_no, item).est

    # Sort items by predicted rating in descending order
    sorted_cf_items = sorted(cf_predictions.keys(), key=lambda x: cf_predictions[x], reverse=True)

    # Get content-based recommendations for each purchased item
    content_based_recommendations = {}
    for item in purchased_items:
        content_based_recommendations[item] = get_content_based_recommendations(item)

    # Combine recommendations from CF and CBF, with relaxed filtering
    hybrid_recommendations = []
    explanations = {}
    for item in sorted_cf_items:
        for rec_item in content_based_recommendations.get(item, []):
            if rec_item not in hybrid_recommendations:
                hybrid_recommendations.append(rec_item)
                explanations[rec_item] = content_based_recommendations[item]

    # Calculate precision and recall
    true_items_set = set(purchased_items)
    recommended_items_set = set(hybrid_recommendations)

    if len(recommended_items_set) == 0:
        precision = 0.0
    else:
        precision = len(true_items_set.intersection(recommended_items_set)) / len(recommended_items_set)

    recall = len(true_items_set.intersection(recommended_items_set)) / len(true_items_set)

    return hybrid_recommendations


# Define an API endpoint to get recommendations
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        transaction_no = request.json['transaction_no']
        recommendations = hybrid_recommendation(transaction_no)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
