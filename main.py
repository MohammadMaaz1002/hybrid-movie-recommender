"""
main.py - Hybrid Movie Recommender (FastAPI)

- Content-based: TF-IDF on genres + overview (cosine similarity)
- Collaborative: Surprise SVD when available; otherwise a simple item-average fallback
- Provides endpoints:
    GET  /                -> health
    GET  /movies          -> list movies (limit query param)
    GET  /recommend/content/{title}?top_n=5 -> content-based recs for a movie title
    GET  /recommend/user/{user_id}?top_n=10 -> hybrid recommendations for a user
    GET  /similar/{movie_id}?top_n=5 -> similar movies by id
Requirements:
    pip install fastapi uvicorn pandas scikit-learn numpy
    Optional (for SVD): pip install scikit-surprise
Datasets expected:
    data/movies.csv  (columns: movieId,title,genres,overview)
    data/ratings.csv (columns: userId,movieId,rating)
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try import Surprise; if missing, we'll fallback
try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except Exception:
    SURPRISE_AVAILABLE = False

DATA_DIR = Path("data")
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"

app = FastAPI(title="Hybrid Movie Recommender", version="1.0")

# ==== Utility data loading ====
def load_movies():
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"{MOVIES_CSV} not found")
    movies = pd.read_csv(MOVIES_CSV)
    # Ensure expected columns
    for col in ("movieId", "title", "genres"):
        if col not in movies.columns:
            raise ValueError(f"movies.csv must contain '{col}' column")
    # Fill overview if missing
    if "overview" not in movies.columns:
        movies["overview"] = ""
    else:
        movies["overview"] = movies["overview"].fillna("")
    return movies

def load_ratings():
    if not RATINGS_CSV.exists():
        # Return empty df if no ratings file present
        return pd.DataFrame(columns=["userId","movieId","rating"])
    ratings = pd.read_csv(RATINGS_CSV)
    return ratings

# ==== Build content-based model ====
class ContentRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        # create a text field combining genres and overview
        self.movies = movies_df.reset_index(drop=True).copy()
        # some cleaning for genres: replace '|' with space
        self.movies["genres_text"] = self.movies["genres"].astype(str).str.replace("|", " ")
        # combine with overview
        self.movies["meta"] = (self.movies["genres_text"] + " " + self.movies["overview"].astype(str)).fillna("")
        # vectorize
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["meta"])
        # Precompute similarity matrix (cosine)
        self.sim_matrix = cosine_similarity(self.tfidf_matrix, dense_output=False)

    def recommend_by_title(self, title: str, top_n: int = 5):
        matches = self.movies[self.movies["title"].str.lower() == title.strip().lower()]
        if matches.empty:
            raise KeyError("title_not_found")
        idx = matches.index[0]
        sims = np.array(self.sim_matrix[idx].toarray()).ravel()
        # get indices sorted by similarity (exclude self)
        top_idxs = sims.argsort()[::-1][1: top_n + 1]
        return self.movies.iloc[top_idxs][["movieId", "title", "genres"]].to_dict(orient="records")

    def similar_by_id(self, movie_id: int, top_n: int = 5):
        row = self.movies[self.movies["movieId"] == movie_id]
        if row.empty:
            raise KeyError("movie_id_not_found")
        idx = row.index[0]
        sims = np.array(self.sim_matrix[idx].toarray()).ravel()
        top_idxs = sims.argsort()[::-1][1: top_n + 1]
        return self.movies.iloc[top_idxs][["movieId", "title", "genres"]].to_dict(orient="records")

# ==== Collaborative / SVD wrapper ====
class CollaborativeRecommender:
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings = ratings_df.copy()
        self.model = None
        self.item_mean = None
        if len(self.ratings) > 0 and SURPRISE_AVAILABLE:
            try:
                reader = Reader(rating_scale=(self.ratings.rating.min(), self.ratings.rating.max()))
                data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)
                trainset = data.build_full_trainset()
                self.model = SVD(n_factors=50, lr_all=0.005, n_epochs=20)
                self.model.fit(trainset)
            except Exception:
                self.model = None
        # fallback: compute item mean rating
        if len(self.ratings) > 0:
            self.item_mean = self.ratings.groupby("movieId")["rating"].mean().to_dict()
        else:
            self.item_mean = {}

    def predict(self, user_id: int, movie_id: int):
        # If SVD available
        if self.model is not None:
            try:
                return self.model.predict(str(user_id), str(movie_id)).est
            except Exception:
                pass
        # fallback to item mean or global mean
        if movie_id in self.item_mean:
            return float(self.item_mean[movie_id])
        if len(self.ratings) > 0:
            return float(self.ratings["rating"].mean())
        return 3.0  # neutral default

# ==== Instantiate models at startup ====
@app.on_event("startup")
def startup_load():
    global movies_df, ratings_df, content_rec, collab_rec
    movies_df = load_movies()
    ratings_df = load_ratings()
    content_rec = ContentRecommender(movies_df)
    collab_rec = CollaborativeRecommender(ratings_df)

# ==== Response models ====
class MovieOut(BaseModel):
    movieId: int
    title: str
    genres: Optional[str] = None

# ==== Routes ====
@app.get("/")
def health():
    return {"status": "ok", "projects": "hybrid-movie-recommender"}

@app.get("/movies", response_model=List[MovieOut])
def list_movies(limit: int = Query(50, ge=1, le=1000)):
    return movies_df[["movieId", "title", "genres"]].head(limit).to_dict(orient="records")

@app.get("/recommend/content/{title}", response_model=List[MovieOut])
def recommend_content(title: str, top_n: int = Query(5, ge=1, le=50)):
    try:
        recs = content_rec.recommend_by_title(title=title, top_n=top_n)
        return recs
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie title '{title}' not found")

@app.get("/similar/{movie_id}", response_model=List[MovieOut])
def similar_movies(movie_id: int, top_n: int = Query(5, ge=1, le=50)):
    try:
        recs = content_rec.similar_by_id(movie_id=movie_id, top_n=top_n)
        return recs
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie id '{movie_id}' not found")

@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, top_n: int = Query(10, ge=1, le=100), alpha: float = Query(0.6, ge=0.0, le=1.0)):
    """
    Hybrid recommendation for a given user_id.
    alpha: weight for collaborative component (0..1). final_score = alpha * collab + (1-alpha) * content
    Strategy:
      - For all candidate movies the user hasn't rated, compute:
          collab_score = predicted rating (SVD or item mean)
          content_score = similarity to user's top-rated movies (avg)
      - Combine and return top_n
    """
    # movies the user rated
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    user_rated_movie_ids = set(user_ratings["movieId"].tolist())

    # If user has no ratings: fallback to top-rated movies by item mean
    candidate_movies = movies_df[~movies_df["movieId"].isin(user_rated_movie_ids)].copy()
    if candidate_movies.empty:
        raise HTTPException(status_code=404, detail="No candidate movies to recommend")

    # Build a content preference vector from user's rated movies (weighted by rating)
    if not user_ratings.empty:
        # take top K rated movies by this user
        top_k = user_ratings.sort_values("rating", ascending=False).head(5)["movieId"].tolist()
        # find their indices in movies_df
        indices = movies_df[movies_df["movieId"].isin(top_k)].index.tolist()
        if indices:
            # aggregate similarity rows
            sim_rows = np.array(content_rec.sim_matrix[indices].toarray())
            # average similarity to all movies
            content_pref = sim_rows.mean(axis=0)  # shape (n_movies,)
        else:
            # no overlap: uniform
            content_pref = np.zeros(len(movies_df))
    else:
        content_pref = np.zeros(len(movies_df))

    # Map movieId -> index in movies_df for quick lookup
    movieid_to_idx = {int(r.movieId): idx for idx, r in movies_df.reset_index().iterrows()}

    scores = []
    for _, row in candidate_movies.iterrows():
        mid = int(row["movieId"])
        # collaborative predicted rating
        collab_score = collab_rec.predict(user_id, mid)
        # content score: similarity to user's preferences; use precomputed content_pref
        idx = movieid_to_idx.get(mid, None)
        content_score = float(content_pref[idx]) if idx is not None else 0.0
        # normalize content_score to roughly 0..5 scale using max of content_pref
        # avoid division by zero
        max_pref = content_pref.max() if content_pref.size else 1.0
        if max_pref > 0:
            content_score_norm = (content_score / max_pref) * 5.0
        else:
            content_score_norm = 0.0
        final_score = float(alpha * collab_score + (1.0 - alpha) * content_score_norm)
        scores.append((mid, final_score))

    # select top_n movieIds by final_score
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    top_movie_ids = [mid for mid, sc in scores_sorted]
    result = movies_df[movies_df["movieId"].isin(top_movie_ids)][["movieId", "title", "genres"]]
    # preserve order
    result = result.set_index("movieId").loc[top_movie_ids].reset_index()
    return {"user_id": user_id, "alpha": alpha, "recommendations": result.to_dict(orient="records")}

# ==== Basic error handling  ====
@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    # keep error messages concise for API consumers
    return {"error": str(exc)}

# ==== If run directly, start uvicorn (optional) ====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
