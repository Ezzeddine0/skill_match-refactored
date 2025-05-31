from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
    return "Service is live!", 200

# Normalize skill (lowercase, trim)
def normalize(skill):
    return skill.lower().strip()

# Partial match percentage + matched skills
def partial_match(query_skills, emp_skills):
    matched_skills = [q for q in query_skills if any(q in e for e in emp_skills)]
    percentage = len(matched_skills) / len(query_skills) if query_skills else 0
    return percentage, matched_skills

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

    # Weights: 80% partial match, 20% embedding similarity
    WEIGHT_PARTIAL = 0.9
    WEIGHT_EMBEDDING = 0.1

    combined_scores = []
    for i, emp in enumerate(employees):
        partial_score, matched_skills = partial_match(query_skills, emp["skills"])
        embedding_score = float(cos_similarities[i])
        combined_score = WEIGHT_PARTIAL * partial_score + WEIGHT_EMBEDDING * embedding_score

        combined_scores.append({
            "id": emp["id"],
            "skills": emp["skills"],
            "matched_skills": matched_skills,
            #"partial_score": partial_score * 100,
            #"embedding_score": embedding_score * 100,
            "combined_score": combined_score * 100,
            #"combined_score_formula": f"{WEIGHT_PARTIAL} * {partial_score*100:.2f} + {WEIGHT_EMBEDDING} * {embedding_score*100:.2f} = {combined_score*100:.2f}"
        })

    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)
    return jsonify(combined_scores)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
