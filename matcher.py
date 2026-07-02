import json
import math
import re
from collections import Counter, defaultdict

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "our", "that", "the", "this", "to", "we",
    "with", "you", "your", "needed", "need", "required", "requiring", "role",
    "job", "intern", "engineer", "developer", "student", "skills", "skill",
}

PHRASE_NORMALIZATION = {
    "machine learning": "machine_learning",
    "deep learning": "deep_learning",
    "natural language processing": "nlp",
    "text classification": "text_classification",
    "data analysis": "data_analysis",
    "data cleaning": "data_cleaning",
    "incident response": "incident_response",
    "risk assessment": "risk_assessment",
    "rest apis": "rest_apis",
    "unit testing": "unit_testing",
    "responsive design": "responsive_design",
    "ci cd": "cicd",
    "ci/cd": "cicd",
}

TOKEN_NORMALIZATION = {
    "js": "javascript",
    "ml": "machine_learning",
    "ai": "artificial_intelligence",
    "apis": "api",
    "dashboards": "dashboard",
    "databases": "database",
    "models": "model",
    "metrics": "metric",
}

CANONICAL_SKILLS = {
    "python": ["python"],
    "sql": ["sql"],
    "excel": ["excel"],
    "pandas": ["pandas"],
    "machine learning": ["machine_learning", "machine learning"],
    "deep learning": ["deep_learning", "deep learning"],
    "neural networks": ["neural networks", "neural_networks"],
    "transformers": ["transformers", "transformer"],
    "nlp": ["nlp", "natural language processing"],
    "tokenization": ["tokenization", "tokenizer"],
    "tf-idf": ["tf-idf", "tfidf"],
    "classification": ["classification", "classifier"],
    "regression": ["regression"],
    "model evaluation": ["model evaluation", "evaluation metric", "metric"],
    "data cleaning": ["data_cleaning", "data cleaning"],
    "data visualization": ["visualization", "charts", "dashboard"],
    "java": ["java"],
    "spring boot": ["spring boot", "spring_boot"],
    "rest api": ["rest_apis", "rest api", "api"],
    "git": ["git"],
    "unit testing": ["unit_testing", "unit testing", "testing"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript"],
    "react": ["react"],
    "responsive design": ["responsive_design", "responsive design"],
    "accessibility": ["accessibility"],
    "linux": ["linux"],
    "network security": ["network security", "network_security"],
    "incident response": ["incident_response", "incident response"],
    "risk assessment": ["risk_assessment", "risk assessment"],
    "siem": ["siem"],
    "aws": ["aws"],
    "docker": ["docker"],
    "ci/cd": ["cicd", "ci cd", "ci/cd"],
    "deployment": ["deployment"],
    "monitoring": ["monitoring"],
    "kotlin": ["kotlin"],
    "android studio": ["android studio", "android_studio"],
    "firebase": ["firebase"],
    "seo": ["seo"],
    "content writing": ["content writing", "content_writing"],
    "analytics": ["analytics"],
    "communication": ["communication"],
    "documentation": ["documentation"],
    "scheduling": ["scheduling"],
    "stakeholder coordination": ["stakeholder", "coordination"],
    "risk tracking": ["risk log", "risk tracking", "risk_logs"],
}


def normalize_text(text):
    text = (text or "").lower().replace("/", " ")
    for phrase, replacement in PHRASE_NORMALIZATION.items():
        text = text.replace(phrase, replacement)
    text = re.sub(r"[^a-z0-9_+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    tokens = re.findall(r"[a-z0-9_+#.]+", normalize_text(text))
    cleaned = []
    for token in tokens:
        token = TOKEN_NORMALIZATION.get(token, token)
        if token not in STOPWORDS and len(token) > 1:
            cleaned.append(token)
    return cleaned


def short_text(text, max_chars=160):
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def extract_skills(text):
    normalized = normalize_text(text)
    padded = f" {normalized} "
    found = set()
    for skill, aliases in CANONICAL_SKILLS.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if f" {alias_norm} " in padded or alias_norm in normalized:
                found.add(skill)
                break
    return sorted(found)


def compare_skills(resume_text, job_description):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_overlap": len(matched) / len(job_skills) if job_skills else 0.0,
    }


def improvement_suggestions(resume_text, job_description, max_items=5):
    comparison = compare_skills(resume_text, job_description)
    suggestions = [
        f"Add evidence of {skill} if you have used it, such as a project, course, or measurable result."
        for skill in comparison["missing_skills"][:max_items]
    ]
    if len(comparison["matched_skills"]) < 3:
        suggestions.append("Make the relevant skills more visible near the top of the CV.")
    if "model evaluation" in comparison["job_skills"] and "model evaluation" not in comparison["resume_skills"]:
        suggestions.append("Mention evaluation metrics such as accuracy, precision, recall, F1-score, or MAE where relevant.")
    if not suggestions:
        suggestions.append("The CV already covers the key requirements. Improve it by adding measurable achievements and project outcomes.")
    return suggestions


class TfidfVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.idf = []

    def fit(self, documents):
        document_frequency = defaultdict(int)
        for document in documents:
            for token in set(tokenize(document)):
                document_frequency[token] += 1
        self.vocabulary = {token: i for i, token in enumerate(sorted(document_frequency))}
        total_docs = max(len(documents), 1)
        self.idf = [
            math.log((1 + total_docs) / (1 + document_frequency[token])) + 1
            for token in sorted(document_frequency)
        ]
        return self

    def transform_one(self, document):
        vector = np.zeros(len(self.vocabulary), dtype=float)
        counts = Counter(tokenize(document))
        if not counts:
            return vector
        max_count = max(counts.values())
        for token, count in counts.items():
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] = (count / max_count) * self.idf[index]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def to_dict(self):
        return {"vocabulary": self.vocabulary, "idf": self.idf}

    @classmethod
    def from_dict(cls, payload):
        vectorizer = cls()
        vectorizer.vocabulary = {k: int(v) for k, v in payload["vocabulary"].items()}
        vectorizer.idf = list(payload["idf"])
        return vectorizer


def cosine_similarity(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


class BowJaccardModel:
    name = "bow_jaccard_baseline"

    def predict_score(self, resume_text, job_description):
        resume_tokens = set(tokenize(resume_text))
        job_tokens = set(tokenize(job_description))
        union = resume_tokens | job_tokens
        return 100.0 * len(resume_tokens & job_tokens) / len(union) if union else 0.0

    def predict(self, resume_text, job_description, threshold=70):
        return build_prediction(self.predict_score(resume_text, job_description), resume_text, job_description, threshold)


class TfidfCosineModel:
    name = "tfidf_cosine"

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer or TfidfVectorizer()

    def fit(self, resumes, jobs):
        self.vectorizer.fit(list(resumes) + list(jobs))
        return self

    def predict_score(self, resume_text, job_description):
        return 100.0 * cosine_similarity(
            self.vectorizer.transform_one(resume_text),
            self.vectorizer.transform_one(job_description),
        )

    def predict(self, resume_text, job_description, threshold=70):
        return build_prediction(self.predict_score(resume_text, job_description), resume_text, job_description, threshold)

    def to_dict(self):
        return {"name": self.name, "vectorizer": self.vectorizer.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        return cls(TfidfVectorizer.from_dict(payload["vectorizer"]))


class HybridMatchingModel:
    name = "hybrid_tfidf_skill"

    def __init__(self, tfidf_model=None, text_weight=0.65, skill_weight=0.35):
        self.tfidf_model = tfidf_model or TfidfCosineModel()
        self.text_weight = text_weight
        self.skill_weight = skill_weight

    def fit(self, resumes, jobs):
        self.tfidf_model.fit(resumes, jobs)
        return self

    def predict_score(self, resume_text, job_description):
        text_score = self.tfidf_model.predict_score(resume_text, job_description)
        skill_score = 100.0 * compare_skills(resume_text, job_description)["skill_overlap"]
        return (self.text_weight * text_score) + (self.skill_weight * skill_score)

    def predict(self, resume_text, job_description, threshold=70):
        return build_prediction(self.predict_score(resume_text, job_description), resume_text, job_description, threshold)

    def to_dict(self):
        return {
            "name": self.name,
            "text_weight": self.text_weight,
            "skill_weight": self.skill_weight,
            "tfidf_model": self.tfidf_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            tfidf_model=TfidfCosineModel.from_dict(payload["tfidf_model"]),
            text_weight=float(payload["text_weight"]),
            skill_weight=float(payload["skill_weight"]),
        )


class LstmSequenceModel:
    name = "lstm_sequence_similarity"

    def __init__(self, embedding_dim=24, hidden_dim=16, max_tokens=120):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(embedding_dim + hidden_dim)
        self.w_i = rng.normal(0, scale, (hidden_dim, embedding_dim + hidden_dim))
        self.w_f = rng.normal(0, scale, (hidden_dim, embedding_dim + hidden_dim))
        self.w_o = rng.normal(0, scale, (hidden_dim, embedding_dim + hidden_dim))
        self.w_g = rng.normal(0, scale, (hidden_dim, embedding_dim + hidden_dim))
        self.b_i = np.zeros(hidden_dim)
        self.b_f = np.ones(hidden_dim) * 0.5
        self.b_o = np.zeros(hidden_dim)
        self.b_g = np.zeros(hidden_dim)

    def token_embedding(self, token):
        seed = sum((index + 1) * ord(char) for index, char in enumerate(token)) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(0, 1, self.embedding_dim)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    @staticmethod
    def sigmoid(vector):
        return 1.0 / (1.0 + np.exp(-vector))

    def encode(self, text):
        tokens = tokenize(text)[: self.max_tokens]
        if not tokens:
            return np.zeros(self.hidden_dim)
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        states = []
        for token in tokens:
            x = self.token_embedding(token)
            combined = np.concatenate([x, h])
            i = self.sigmoid(self.w_i @ combined + self.b_i)
            f = self.sigmoid(self.w_f @ combined + self.b_f)
            o = self.sigmoid(self.w_o @ combined + self.b_o)
            g = np.tanh(self.w_g @ combined + self.b_g)
            c = (f * c) + (i * g)
            h = o * np.tanh(c)
            states.append(h)
        vector = np.mean(states, axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def predict_score(self, resume_text, job_description):
        return 100.0 * cosine_similarity(self.encode(resume_text), self.encode(job_description))

    def predict(self, resume_text, job_description, threshold=70):
        return build_prediction(self.predict_score(resume_text, job_description), resume_text, job_description, threshold)


def build_prediction(score, resume_text, job_description, threshold=70):
    skills = compare_skills(resume_text, job_description)
    return {
        "match_score": round(float(score), 2),
        "prediction": "Fit" if score >= threshold else "Not Fit",
        "threshold": threshold,
        "matched_skills": skills["matched_skills"],
        "missing_skills": skills["missing_skills"],
        "resume_skills": skills["resume_skills"],
        "job_skills": skills["job_skills"],
        "skill_overlap": round(skills["skill_overlap"], 3),
        "suggestions": improvement_suggestions(resume_text, job_description),
    }


def classification_metrics(y_true, y_pred):
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": round((tp + tn) / total, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def mean_absolute_error(y_true_scores, y_pred_scores):
    return round(sum(abs(a - b) for a, b in zip(y_true_scores, y_pred_scores)) / len(y_true_scores), 3) if y_true_scores else 0.0


def root_mean_squared_error(y_true_scores, y_pred_scores):
    if not y_true_scores:
        return 0.0
    return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true_scores, y_pred_scores)) / len(y_true_scores)), 3)


def save_hybrid_model(model, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model.to_dict(), file, indent=2)


def load_hybrid_model(path):
    with open(path, "r", encoding="utf-8") as file:
        return HybridMatchingModel.from_dict(json.load(file))
