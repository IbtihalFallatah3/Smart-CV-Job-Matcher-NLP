import argparse
import json
from pathlib import Path

from matcher import HybridMatchingModel, load_hybrid_model


def read_text(value, path):
    if path:
        return Path(path).read_text(encoding="utf-8")
    return value or ""


def main():
    parser = argparse.ArgumentParser(description="Predict CV-job match score.")
    parser.add_argument("--resume", default="")
    parser.add_argument("--job", default="")
    parser.add_argument("--resume-file")
    parser.add_argument("--job-file")
    parser.add_argument("--model-path", default="hybrid_model.json")
    parser.add_argument("--threshold", type=float, default=70.0)
    args = parser.parse_args()

    resume_text = read_text(args.resume, args.resume_file)
    job_description = read_text(args.job, args.job_file)

    if Path(args.model_path).exists():
        model = load_hybrid_model(args.model_path)
    else:
        model = HybridMatchingModel()
        model.fit([resume_text], [job_description])

    prediction = model.predict(resume_text, job_description, threshold=args.threshold)
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
