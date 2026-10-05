import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)



app = FastAPI()

def get_test_results():
    
    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    df_test = pd.read_csv("BANK LOAN_test.csv")

    X_val = df_test.drop(["DEFAULTER", "SN"], axis=1)


    y_proba = model.predict_proba(X_val)[:,1]


    X_val["Default_Probability_%"] = np.round(y_proba*100,2)

    return(X_val)

@app.get("/predict")
def predict():
    try:
        df = get_test_results()

        # Handle invalid values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Return JSON
        return df.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def home():

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Loan Default Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
            }
            button {
                padding: 10px 16px;
                font-size: 16px;
                margin-bottom: 20px;
                cursor: pointer;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f4f4f4;
            }
        </style>
    </head>
    <body>

        <h2>Loan Default Prediction</h2>

        <button onclick="loadData()">Load Summary</button>

        <table id="DataTable">
            <thead>
                <tr>
                    <th>AGE</th>
                    <th>EMPLOY</th>
                    <th>ADDRESS</th>
                    <th>DEBTINC</th>
                    <th>CREDDEBT</th>
                    <th>OTHDEBT</th>
                    <th>Default_Probability (%)</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            function loadData() {
                fetch('/predict')
                    .then(response => response.json())
                    .then(data => {
                        const tbody = document.querySelector('#DataTable tbody');
                        tbody.innerHTML = '';

                        data.forEach(row => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${row.AGE}</td>
                                <td>${row.EMPLOY}</td>
                                <td>${row.ADDRESS}</td>
                                <td>${row.DEBTINC}</td>
                                <td>${row.CREDDEBT}</td>
                                <td>${row.OTHDEBT}</td>
                                <td>${row["Default_Probability_%"]}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    })
                    .catch(error => {
                        alert('Error fetching data');
                        console.error(error);
                    });
            }
        </script>

    </body>
    </html>
    """

    return html_content
    
@app.post("/predict_single")
async def predict_single(request: Request):

    try:
        data = await request.json()

        X = pd.DataFrame([{
            "AGE": data["AGE"],
            "EMPLOY": data["EMPLOY"],
            "ADDRESS": data["ADDRESS"],
            "DEBTINC": data["DEBTINC"],
            "CREDDEBT": data["CREDDEBT"],
            "OTHDEBT": data["OTHDEBT"]
        }])

        probability = model.predict_proba(X)[0, 1]

        return {
            "default_probability": round(probability * 100, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }