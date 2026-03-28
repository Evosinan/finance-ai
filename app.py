from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

app = FastAPI()

# -------------------------
# CONFIG
# -------------------------
admin_email = "admin@financeai.com"
USER_FILE = "users.json"

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DATA MODELS
# -------------------------
class GenerateRequest(BaseModel):
    prompt: str
    email: str

class Feedback(BaseModel):
    usefulness: str
    rating: str
    comment: str

class Business(BaseModel):
    objective: str
    pain: str

class User(BaseModel):
    email: str

class AnalysisRequest(BaseModel):
    email: str
    income: float
    expenses: float
    goal: str

# -------------------------
# FILE HELPERS
# -------------------------
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------
# LOAD MODEL
# -------------------------
def fake_ai_response(prompt):
    return f"AI Response (demo): {prompt}"

@app.post("/generate")
def generate_text(request: GenerateRequest):
    return {"response": fake_ai_response(request.prompt)}

# -------------------------
# LOGIN
# -------------------------
@app.post("/login")
def login(user: User):
    users = load_users()

    if user.email not in users:
        users[user.email] = {"usage": 0}

    save_users(users)

    return {"status": "logged_in"}

# -------------------------
# GENERATE (AI CHAT)
# -------------------------
@app.post("/generate")
def generate_text(request: GenerateRequest):
    users = load_users()

    if request.email not in users:
        users[request.email] = {"usage": 0}

    if users[request.email]["usage"] >= 5:
        return {"response": "Free limit reached. Upgrade to Pro."}

    users[request.email]["usage"] += 1
    save_users(users)

    prompt = f"""
You are a professional financial advisor AI.

STRICT RULES:
- ONLY answer finance, business, or investment questions
- If NOT finance-related, say: "This question is outside my domain."
- Keep answers short and complete

Question: {request.prompt}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in response:
        response = response.split("Answer:")[1]

    return {"response": response.strip()}

# -------------------------
# ANALYSIS (PRO FEATURE)
# -------------------------
@app.post("/analysis")
def analyze_finance(data: AnalysisRequest):
    users = load_users()

    if data.email not in users:
        users[data.email] = {"usage": 0}

    if users[data.email]["usage"] >= 5:
        return {"response": "Upgrade required for advanced analysis."}

    users[data.email]["usage"] += 1
    save_users(users)

    prompt = f"""
You are an expert financial advisor.

Analyze the user's finances:

Income: {data.income}
Expenses: {data.expenses}
Goal: {data.goal}

Provide:
1. Financial health
2. Risk level
3. Step-by-step plan
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response.strip()}

# -------------------------
# FEEDBACK
# -------------------------
@app.post("/feedback")
def save_feedback(data: Feedback):
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"{data.usefulness} | {data.rating} | {data.comment}\n")

    return {"status": "saved"}

# -------------------------
# BUSINESS
# -------------------------
@app.post("/business")
def save_business(data: Business):
    with open("business.txt", "a", encoding="utf-8") as f:
        f.write(f"Objective: {data.objective} | Pain: {data.pain}\n")

    return {"status": "saved"}

# -------------------------
# ADMIN DASHBOARD
# -------------------------
@app.get("/admin")
def admin_dashboard(email: str):

    if email != admin_email:
        return {"error": "Unauthorized"}

    import json

    # Load users
    try:
        with open("users.json", "r") as f:
            users_data = json.load(f)
    except:
        users_data = {}

    # Load feedback
    try:
        with open("feedback.txt", "r", encoding="utf-8") as f:
            feedback_data = f.readlines()
    except:
        feedback_data = []

    # Load business
    try:
        with open("business.txt", "r", encoding="utf-8") as f:
            business_data = f.readlines()
    except:
        business_data = []

    return {
        "total_users": len(users_data),
        "users": users_data,
        "feedback": feedback_data[-10:],
        "business": business_data[-10:]
    }