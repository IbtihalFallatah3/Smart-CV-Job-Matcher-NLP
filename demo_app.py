import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from matcher import HybridMatchingModel, load_hybrid_model


DEFAULT_RESUME = """AI student with Python, NLP, TF-IDF, text classification,
tokenization, data cleaning, model evaluation, and dashboard experience."""

DEFAULT_JOB = """NLP intern needed with Python, text preprocessing, tokenization,
TF-IDF, classification, evaluation metrics, transformers, and clear reporting."""


def load_model(model_path):
    path = Path(model_path)
    if path.exists():
        return load_hybrid_model(path)
    model = HybridMatchingModel()
    model.fit([DEFAULT_RESUME], [DEFAULT_JOB])
    return model


def render_list(items):
    if not items:
        return "<li>None</li>"
    return "".join(f"<li>{html.escape(str(item))}</li>" for item in items)


def page(prediction=None, resume_text=DEFAULT_RESUME, job_description=DEFAULT_JOB):
    result_html = ""
    if prediction:
        score = prediction["match_score"]
        result_html = f"""
        <section class="result">
          <div class="score">
            <span>{score:.2f}%</span>
            <strong>{html.escape(prediction["prediction"])}</strong>
          </div>
          <div class="grid">
            <article>
              <h2>Matched skills</h2>
              <ul>{render_list(prediction["matched_skills"])}</ul>
            </article>
            <article>
              <h2>Missing skills</h2>
              <ul>{render_list(prediction["missing_skills"])}</ul>
            </article>
            <article class="wide">
              <h2>CV improvement suggestions</h2>
              <ul>{render_list(prediction["suggestions"])}</ul>
            </article>
          </div>
        </section>
        """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart CV-Job Matcher</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18202a;
      --muted: #586272;
      --line: #d6dbe3;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.5;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 22px 28px;
    }}
    header h1 {{
      margin: 0 0 4px;
      font-size: 26px;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
      max-width: 900px;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 24px auto 40px;
    }}
    form {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: start;
    }}
    label {{
      display: grid;
      gap: 8px;
      font-weight: 700;
    }}
    textarea {{
      width: 100%;
      min-height: 260px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font: 15px/1.5 Arial, Helvetica, sans-serif;
      background: var(--panel);
      color: var(--ink);
    }}
    .actions {{
      grid-column: 1 / -1;
      display: flex;
      justify-content: flex-end;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      padding: 12px 18px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .result {{
      margin-top: 24px;
      display: grid;
      gap: 18px;
    }}
    .score {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      border: 1px solid var(--line);
      border-left: 6px solid var(--accent);
      background: var(--panel);
      border-radius: 8px;
      padding: 18px;
    }}
    .score span {{
      font-size: 38px;
      font-weight: 800;
    }}
    .score strong {{
      font-size: 20px;
      color: var(--accent-strong);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    article {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 18px;
    }}
    article.wide {{ grid-column: 1 / -1; }}
    h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    ul {{ margin: 0; padding-left: 20px; }}
    li + li {{ margin-top: 6px; }}
    @media (max-width: 760px) {{
      form, .grid {{ grid-template-columns: 1fr; }}
      .score {{ display: grid; gap: 4px; }}
      article.wide {{ grid-column: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Smart CV-Job Matcher</h1>
    <p>Paste a CV and job description to predict fit, identify matched and missing skills, and generate improvement suggestions.</p>
  </header>
  <main>
    <form method="post">
      <label>Resume / CV
        <textarea name="resume_text">{html.escape(resume_text)}</textarea>
      </label>
      <label>Job description
        <textarea name="job_description">{html.escape(job_description)}</textarea>
      </label>
      <div class="actions">
        <button type="submit">Analyze match</button>
      </div>
    </form>
    {result_html}
  </main>
</body>
</html>"""


class DemoHandler(BaseHTTPRequestHandler):
    model = None

    def do_GET(self):
        self.respond(page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        values = parse_qs(body)
        resume_text = values.get("resume_text", [""])[0]
        job_description = values.get("job_description", [""])[0]
        prediction = self.model.predict(resume_text, job_description)
        self.respond(page(prediction, resume_text, job_description))

    def respond(self, body, status=200, content_type="text/html; charset=utf-8"):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        print(json.dumps({"client": self.address_string(), "message": format % args}))


def main():
    parser = argparse.ArgumentParser(description="Run the local CV-job matching demo.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default="hybrid_model.json")
    args = parser.parse_args()

    DemoHandler.model = load_model(args.model_path)
    server = HTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
