import argparse
import csv
from pathlib import Path

from matcher import (
    BowJaccardModel,
    HybridMatchingModel,
    LstmSequenceModel,
    TfidfCosineModel,
    classification_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    save_hybrid_model,
    short_text,
)


def read_dataset(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def split_dataset(rows, test_ratio=0.3):
    import random

    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    test = shuffled[:test_size]
    train = shuffled[test_size:]
    return train, test


def evaluate_model(model, dataset_rows, threshold=70):
    true_scores = [float(row["match_score"]) for row in dataset_rows]
    true_labels = [1 if score >= threshold else 0 for score in true_scores]
    predicted_scores = []
    predicted_labels = []
    detail_rows = []

    for row in dataset_rows:
        prediction = model.predict(row["resume_text"], row["job_description"], threshold=threshold)
        predicted_score = prediction["match_score"]
        predicted_scores.append(predicted_score)
        predicted_labels.append(1 if predicted_score >= threshold else 0)
        detail_rows.append(
            {
                "resume_preview": short_text(row["resume_text"]),
                "job_preview": short_text(row["job_description"]),
                "actual_score": float(row["match_score"]),
                "predicted_score": predicted_score,
                "actual_label": "Fit" if float(row["match_score"]) >= threshold else "Not Fit",
                "predicted_label": prediction["prediction"],
                "matched_skills": ", ".join(prediction["matched_skills"]),
                "missing_skills": ", ".join(prediction["missing_skills"]),
            }
        )

    metrics = classification_metrics(true_labels, predicted_labels)
    metrics["mae"] = mean_absolute_error(true_scores, predicted_scores)
    metrics["rmse"] = root_mean_squared_error(true_scores, predicted_scores)
    return metrics, detail_rows


def write_metrics(path, results):
    fieldnames = ["model", "accuracy", "precision", "recall", "f1", "mae", "rmse", "tp", "tn", "fp", "fn"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow({"model": model_name, **metrics})


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CV-job matching models.")
    parser.add_argument("--data", default="sample_resume_job_matches.csv")
    parser.add_argument("--model-path", default="hybrid_model.json")
    parser.add_argument("--threshold", type=float, default=70.0)
    args = parser.parse_args()

    model_path = Path(args.model_path)

    rows = read_dataset(args.data)
    train_rows, test_rows = split_dataset(rows)

    bow_model = BowJaccardModel()
    train_resumes = [row["resume_text"] for row in train_rows]
    train_jobs = [row["job_description"] for row in train_rows]
    tfidf_model = TfidfCosineModel().fit(train_resumes, train_jobs)
    hybrid_model = HybridMatchingModel().fit(train_resumes, train_jobs)
    lstm_model = LstmSequenceModel()

    models = [bow_model, tfidf_model, hybrid_model, lstm_model]
    results = {}
    detail_rows = []
    for model in models:
        metrics, rows = evaluate_model(model, test_rows, threshold=args.threshold)
        results[model.name] = metrics
        for row in rows:
            detail_rows.append({"model": model.name, **row})

    write_metrics("metrics.csv", results)
    save_hybrid_model(hybrid_model, model_path)

    print("Training complete.")
    print(f"Train samples: {len(train_rows)}")
    print(f"Test samples: {len(test_rows)}")
    for name, metrics in results.items():
        print(f"{name}: accuracy={metrics['accuracy']} f1={metrics['f1']} mae={metrics['mae']}")
    print(f"Saved model: {model_path}")
    print("Saved metrics: metrics.csv")


if __name__ == "__main__":
    main()
