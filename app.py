from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here_change_in_production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_pickle("job_data (1).pkl")
job_embeddings = np.load("job_embeddings (1).npy")

def match_jobs(resume_text):
    resume_embedding = model.encode([resume_text])
    similarities = cosine_similarity(resume_embedding, job_embeddings)[0] * 100
    df['match_score'] = similarities

    top_jobs = df.sort_values(by="match_score", ascending=False).head(5)

    results = []
    for _, row in top_jobs.iterrows():
        organization = row['organization'] if pd.notna(row['organization']) else "Confidential Employer"
        results.append({
            "title": row['job_title'],
            "location": row['location'],
            "organization": organization,
            "score": round(row['match_score'], 2)
        })

    return results


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    results = None

    if request.method == "POST":

        resume_text = request.form.get("resume")

        # Check if file uploaded
        if "resume_file" in request.files:
            file = request.files["resume_file"]
            if file.filename != "":
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                resume_text = text

        if resume_text:
            results = match_jobs(resume_text)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)