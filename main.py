'''
Important!
Enter your full name (as it appears on Canvas) and NetID.  
If you are working in a group (maximum of 4 members), include the full names and NetIDs of all your partners.  
If you're working alone, enter `None` for the partner fields.
'''

'''
Project: MP5
Student 1: Sofia Roth, siroth
Student 2: None
'''
# Dataset source: Sleep Health and Lifestyle Dataset from Kaggle
# https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

import io
import re
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import flask
from flask import Flask, jsonify, make_response, request, Response
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


app = Flask(__name__)
DATA_PATH = "data/main.csv"
df = pd.read_csv(DATA_PATH)

home_visit_count = 0
donate_from_a = 0
donate_from_b = 0


def compute_summary(data: pd.DataFrame) -> dict:
    """Compute simple metrics and findings for homepage + /summary.json."""
    out = {}

    out["n_rows"] = int(len(data))
    out["avg_sleep_duration"] = float(data["Sleep Duration"].mean()) if "Sleep Duration" in data else None
    out["avg_sleep_quality"] = float(data["Quality of Sleep"].mean()) if "Quality of Sleep" in data else None

    # Define "poor sleep" threshold (you can adjust)
    if "Quality of Sleep" in data:
        poor = (data["Quality of Sleep"] <= 5).mean()
        out["poor_sleep_rate"] = float(poor)

    # Top occupations by mean sleep quality
    if "Occupation" in data and "Quality of Sleep" in data:
        occ = (
            data.groupby("Occupation")["Quality of Sleep"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
        )
        out["top_occupations_by_quality"] = [
            {"occupation": str(k), "mean_quality": float(v)} for k, v in occ.items()
        ]

    # Quick correlation-ish signal
    if "Physical Activity Level" in data and "Quality of Sleep" in data:
        corr = data["Physical Activity Level"].corr(data["Quality of Sleep"])
        out["corr_activity_quality"] = float(corr)

    return out


def train_models(data: pd.DataFrame) -> dict:
    """
    Train:
    - Regression: predict Quality of Sleep
    - Classification: predict poor sleep (Quality <= 5)
    Returns metrics and top coefficients.
    """
    # Features available in this dataset
    candidate_features = [
        "Age",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Sleep Duration",
    ]
    features = [c for c in candidate_features if c in data.columns]
    if "Quality of Sleep" not in data.columns or len(features) == 0:
        return {"error": "Missing required columns for modeling."}

    model_df = data.dropna(subset=features + ["Quality of Sleep"]).copy()

    X = model_df[features].values
    y = model_df["Quality of Sleep"].values

    # --- Regression ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    reg_coefs = sorted(
        [{"feature": f, "coef": float(c)} for f, c in zip(features, reg.coef_)],
        key=lambda d: abs(d["coef"]),
        reverse=True,
    )

    # --- Classification: poor sleep vs not ---
    y_bin = (model_df["Quality of Sleep"] <= 5).astype(int).values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_bin, test_size=0.25, random_state=42)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train2, y_train2)
    prob = clf.predict_proba(X_test2)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_test2, pred)
    try:
        auc = roc_auc_score(y_test2, prob)
    except Exception:
        auc = None

    clf_coefs = sorted(
        [{"feature": f, "coef": float(c)} for f, c in zip(features, clf.coef_[0])],
        key=lambda d: abs(d["coef"]),
        reverse=True,
    )

    return {
        "features_used": features,
        "regression": {
            "target": "Quality of Sleep",
            "model": "LinearRegression",
            "r2": float(r2),
            "top_coefficients": reg_coefs[:3],
        },
        "classification": {
            "target": "Poor sleep (Quality <= 5)",
            "model": "LogisticRegression",
            "accuracy": float(acc),
            "auc": float(auc) if auc is not None else None,
            "top_coefficients": clf_coefs[:3],
        },
    }


SUMMARY = compute_summary(df)
MODEL_RESULTS = train_models(df)


@app.route('/')
def home():
    global home_visit_count
    home_visit_count += 1
    with open("index.html") as f:
        html = f.read()
    if home_visit_count <= 10:
        if home_visit_count % 2 == 1:
            html = html.replace("donate.html", "donate.html?from=A")
            html = html.replace("{{DONATE_COLOR}}", "blue")
        else:
            html = html.replace("donate.html", "donate.html?from=B")
            html = html.replace("{{DONATE_COLOR}}", "red")
    else:
        if donate_from_a >= donate_from_b:
            html = html.replace("donate.html", "donate.html?from=A")
            html = html.replace("{{DONATE_COLOR}}", "blue")
        else:
            html = html.replace("donate.html", "donate.html?from=B")
            html = html.replace("{{DONATE_COLOR}}", "red")
            
    findings = []
    if SUMMARY.get("corr_activity_quality") is not None:
        findings.append(f"Physical activity correlates with sleep quality (corr ≈ {SUMMARY['corr_activity_quality']:.2f}).")
    if SUMMARY.get("avg_sleep_quality") is not None:
        findings.append(f"Average sleep quality is {SUMMARY['avg_sleep_quality']:.2f} (scale 1–10).")
    if SUMMARY.get("poor_sleep_rate") is not None:
        findings.append(f"{SUMMARY['poor_sleep_rate']*100:.1f}% of records meet the 'poor sleep' threshold (≤ 5).")

    top_occ = SUMMARY.get("top_occupations_by_quality", [])
    if top_occ:
        occ_str = ", ".join([f"{x['occupation']} ({x['mean_quality']:.2f})" for x in top_occ])
        findings.append(f"Top occupations by mean sleep quality: {occ_str}.")

    findings_html = "<ul>" + "".join([f"<li>{f}</li>" for f in findings[:4]]) + "</ul>"

    html = html.replace("{{KEY_FINDINGS}}", findings_html)
    html = html.replace("{{N_ROWS}}", str(SUMMARY.get("n_rows", "")))

    return html


@app.route("/summary.json")
def summary_json():
    return jsonify(SUMMARY)


@app.route("/model.json")
def model_json():
    return jsonify(MODEL_RESULTS)


@app.route('/browse.html')
def browse_html():
    table_html = df.to_html(index=False, border=1, justify="center")
    return f"""
    <html>
        <head>
            <title>Browse Sleep Dataset</title>
            <style>
                body {{
                    font-family: Roboto, serif;
                    background-color: #fafafa;
                    padding: 20px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <a href="/">Back to Home</a>
            <h1>Browse Dataset</h1>
            {table_html}
        </body>
    </html>"""

last_request_time = {}
@app.route('/browse.json')
def browse_json():
    client_ip = request.remote_addr
    current_time = time.time()
    
    if client_ip in last_request_time:
        elapsed = current_time - last_request_time[client_ip]
        if elapsed < 60:  # less than 60 seconds
            retry_after = 60 - int(elapsed)
            response = make_response("Too Many Requests", 429)
            response.headers['Retry-After'] = str(retry_after)
            return response

    last_request_time[client_ip] = current_time
    
    gender = request.args.get("gender")
    occupation = request.args.get("occupation")

    data = df
    if gender:
        data = data[data["Gender"].astype(str).str.upper() == gender.upper()]
    if occupation:
        data = data[data["Occupation"].astype(str).str.lower() == occupation.lower()]

    return jsonify(data.to_dict(orient="records"))

@app.route('/visitors.json')
def log():
    ip_list = list(last_request_time.keys())
    return jsonify(ip_list)

@app.route('/donate.html')
def donate():
    global donate_from_a, donate_from_b
    version = request.args.get("from")
    if version == "A":
        donate_from_a += 1
    elif version == "B":
        donate_from_b += 1
    return f"""
    <html>
        <head>
            <title>Donations</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #fafafa;
                    padding: 20px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <a href="/">Back to Home</a>
            <h1>Donate to Sleep Data</h1>
            <p>
                Donations toward funding the research sleep data is contributing to our future of health.<br>
                Sleep is essential to human health, and with more knowledge around sleep and sleeping patterns we<br>
                will be able to create sleep aids to improve struggles surrounding sleep. Sleep should be a priority <br>
                help us make it just that!
            </p>
        </body>
    </html>
    """
@app.route('/email', methods=["POST"])
def email():
    email_text = str(request.data, "utf-8").strip()
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$"
    if re.match(pattern, email_text):
        with open("emails.txt", "a") as f:
            f.write(email_text + "\n")
        num_subscribed = sum(1 for _ in open("emails.txt"))
        return jsonify(f"Thanks! Your subscriber number is {num_subscribed}.")

    return jsonify("Invalid email address.")

@app.route("/dashboard1.svg")
def activity():
    gender = request.args.get("gender")
    min_activity = request.args.get("min_activity", type=float)
    max_activity = request.args.get("max_activity", type=float)

    # ✅ define data
    data = df.copy()

    # ✅ apply filters
    if gender:
        data = data[data["Gender"].astype(str).str.upper() == gender.upper()]

    if min_activity is not None:
        data = data[data["Physical Activity Level"] >= min_activity]
    if max_activity is not None:
        data = data[data["Physical Activity Level"] <= max_activity]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    female_data = data[data["Gender"] == "Female"]
    male_data = data[data["Gender"] == "Male"]

    if not female_data.empty:
        ax.scatter(
            female_data["Physical Activity Level"],
            female_data["Quality of Sleep"],
            alpha=0.55,
            s=28,
            label="Female"
        )

    if not male_data.empty:
        ax.scatter(
            male_data["Physical Activity Level"],
            male_data["Quality of Sleep"],
            alpha=0.55,
            s=28,
            label="Male"
        )

    x = data["Physical Activity Level"].to_numpy()
    y = data["Quality of Sleep"].to_numpy()

    if len(x) > 1 and np.nanstd(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, slope * xs + intercept, linewidth=2, label="Trend")

    ax.set_title("Sleep Quality vs Physical Activity", fontsize=13)
    ax.set_xlabel("Physical Activity Level", fontsize=11)
    ax.set_ylabel("Sleep Quality (1–10)", fontsize=11)
    ax.set_ylim(0.5, 10.5)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return flask.Response(buf.getvalue(), headers={"Content-type": "image/svg+xml"})

@app.route("/dashboard2.svg")
def occupation_activity():
    top_n = request.args.get("top_n", default=10, type=int)
    occupation = request.args.get("occupation")

    # define data
    data = df.copy()

    # optional filter by occupation
    if occupation:
        data = data[data["Occupation"].astype(str).str.lower() == occupation.lower()]

    mean_quality = (
        data.groupby("Occupation")["Quality of Sleep"]
        .mean()
        .sort_values(ascending=True)
    )

    if top_n and top_n > 0:
        mean_quality = mean_quality.tail(top_n)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(mean_quality.index, mean_quality.values)

    ax.set_title(f"Top {len(mean_quality)} Occupations by Mean Sleep Quality", fontsize=13)
    ax.set_xlabel("Mean Sleep Quality (1–10)", fontsize=11)
    ax.set_ylabel("Occupation", fontsize=11)

    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return flask.Response(buf.getvalue(), headers={"Content-type": "image/svg+xml"})

@app.route("/charts")
def charts():
    return """
    <html>
      <head>
        <title>Sleep Dashboard Charts</title>
        <style>
          body { font-family: Arial, sans-serif; background:#f2f2f2; padding:20px; text-align:center; }
          .card { background:#fff; max-width: 950px; margin: 0 auto 18px auto; padding:16px; border-radius:12px; }
          img { width: 100%; height: auto; }
          a { color:#005776; text-decoration:none; }
          label { margin: 0 10px; }
        </style>
      </head>
      <body>
        <div class="card">
          <a href="/">← Back to Home</a>
          <h1 style="color:#003E51;">Charts</h1>
          <p>Use the filters to update the graphs.</p>

          <div style="margin: 12px 0;">
            <label>
              Gender:
              <select id="gender">
                <option value="">All</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </label>

            <label>
              Min Activity:
              <input id="minA" type="number" value="0" min="0" max="100" style="width:70px;">
            </label>

            <label>
              Max Activity:
              <input id="maxA" type="number" value="100" min="0" max="100" style="width:70px;">
            </label>

            <button onclick="apply()">Apply</button>
          </div>
        </div>

        <div class="card">
          <h2 style="color:#003E51;">Sleep Quality vs Physical Activity</h2>
          <img id="chart1" src="/dashboard1.svg">
          <p><a id="open1" href="/dashboard1.svg" target="_blank">Open this chart alone</a></p>
        </div>

        <div class="card">
          <h2 style="color:#003E51;">Top Occupations by Mean Sleep Quality</h2>
          <img id="chart2" src="/dashboard2.svg?top_n=10">
          <p><a id="open2" href="/dashboard2.svg?top_n=10" target="_blank">Open this chart alone</a></p>
        </div>

        <script>
          function apply() {
            const g = document.getElementById("gender").value;
            const minA = document.getElementById("minA").value;
            const maxA = document.getElementById("maxA").value;

            const params = new URLSearchParams();
            if (g) params.set("gender", g);
            if (minA !== "") params.set("min_activity", minA);
            if (maxA !== "") params.set("max_activity", maxA);

            // cache buster
            params.set("_t", Date.now());

            const url1 = "/dashboard1.svg?" + params.toString();
            document.getElementById("chart1").src = url1;
            document.getElementById("open1").href = url1;

            // chart2 stays same for now (or you can add occupation/top_n controls later)
          }
        </script>
      </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, threaded=False)


# Add implementation here
