# 🎬 Hybrid Movie Recommender System

> **An intelligent hybrid movie recommendation web app built with FastAPI, combining Content-Based Filtering (TF-IDF) and Collaborative Filtering (SVD) to deliver personalized movie suggestions.**  
> Developed by **Mohammad Maaz**.

---

## 📘 Overview

The **Hybrid Movie Recommender System** merges two powerful recommendation strategies:

1. **Content-Based Filtering** — uses TF-IDF vectorization on movie genres and overviews to find similar movies.  
2. **Collaborative Filtering** — applies Singular Value Decomposition (SVD) on user-movie rating matrices to predict unseen ratings.

This hybrid approach ensures accurate, diverse, and personalized movie recommendations.

---

## 🚀 Features

- 🔐 **User Authentication:** Sign-up and login with preferred genres.  
- 🎞️ **Personalized Recommendations:** Based on both your interests and other users’ tastes.  
- 🔍 **Search Functionality:** Quickly find movies by title or genre.  
- 🧠 **Hybrid Engine:** Combines TF-IDF + SVD for enhanced performance.  
- 📊 **Data Visualization:** Displays genre distribution, rating trends, and similarity scores.  
- 🌐 **FastAPI Backend:** Lightweight, asynchronous, and production-ready.  

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Input / Sign-Up] --> B[Content-Based Filtering]
    A --> C[Collaborative Filtering]
    B --> D[Hybrid Engine]
    C --> D
    D --> E[FastAPI Backend]
    E --> F[Frontend: HTML & Jinja Templates]
    F --> G[Personalized Movie Recommendations]

