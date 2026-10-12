import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("movie_recommendation_dataset.csv")

print("Script started")


popular_genres = df.groupby('Genre')['Rating'].mean().sort_values(ascending=False)
print("⭐ Most Popular Genres (Highest Avg Ratings):\n", popular_genres)


best_genre = popular_genres.idxmax()
worst_genre = popular_genres.idxmin()
print(f"\n✅ Best Genre: {best_genre}")
print(f"❌ Worst Genre: {worst_genre}")

df['features'] = df['Genre'] + " " + df['MovieName']


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_movie(movie_name, df, cosine_sim):
    if movie_name not in df['MovieName'].values:
        return "Movie not found in the database."
    idx = df[df['MovieName'] == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] 
    movie_indices = [i[0] for i in sim_scores]
    return df['MovieName'].iloc[movie_indices]

print("\n🎬 Recommended Movies like 'Inception':")
print(recommend_movie('Inception', df, cosine_sim))
