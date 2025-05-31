from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Normalize skill (lowercase, trim)
def normalize(skill):
    return skill.lower().strip()

# Partial match percentage
def partial_match_percentage(query_skills, emp_skills):
    matched_skills = [q for q in query_skills if any(q in e for e in emp_skills)]
    return len(matched_skills) / len(query_skills) if query_skills else 0

@app.route("/match", methods=["POST"])
def match_employees():
    data = request.get_json()
    employees = data["employees"]
    query_skills = [normalize(s) for s in data["query_skills"]]

    # Prepare employee skill text
    for emp in employees:
        emp["skills"] = [normalize(s) for s in emp["skills"]]
        emp["skills_text"] = ", ".join(emp["skills"])

    # Vectorize with TF-IDF
    texts = [", ".join(query_skills)] + [emp["skills_text"] for emp in employees]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity
    query_vector = tfidf_matrix[0]
    employee_vectors = tfidf_matrix[1:]
    cos_similarities = cosine_similarity(query_vector, employee_vectors)[0]

    # Weights
    WEIGHT_EMBEDDING = 0.7
    WEIGHT_PARTIAL = 0.3

    combined_scores = []
    for i, emp in enumerate(employees):
        partial_score = partial_match_percentage(query_skills, emp["skills"])
        embedding_score = float(cos_similarities[i])
        combined_score = WEIGHT_EMBEDDING * embedding_score + WEIGHT_PARTIAL * partial_score
        combined_scores.append({
            "id": emp["id"],
            "skills": emp["skills"],
            "combined_score": combined_score * 100
        })

    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)
    return jsonify(combined_scores)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
