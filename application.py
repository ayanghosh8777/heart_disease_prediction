import pickle
from flask import Flask, render_template, request
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

app.config['SECRET_KEY'] = 'your-secret-key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = 'login'

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )
from datetime import datetime
class PredictionHistory(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )

    risk_percentage = db.Column(
        db.Float,
        nullable=False
    )

    risk_level = db.Column(
        db.String(50),
        nullable=False
    )

    result = db.Column(
        db.String(100),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )
with app.app_context():
      db.create_all()
# Load model and scaler
best_model = pickle.load(
    open("models/new_best_model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)
@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))

@app.route("/")
def index():
    return render_template("index.html")
@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        username = request.form.get('username')

        email = request.form.get('email')

        password = request.form.get('password')

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            return "Email already registered"

        hashed_password = generate_password_hash(
            password
        )

        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)

        db.session.commit()

        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form.get('email')

        password = request.form.get('password')

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            login_user(user)

            return redirect(
                url_for('predict_disease')
            )

        return "Invalid email or password"

    return render_template('login.html')
@app.route('/logout')
@login_required
def logout():

    logout_user()

    return redirect(
        url_for('login')
    )
@app.route('/history')
@login_required
def history():

    predictions = PredictionHistory.query.filter_by(

        user_id=current_user.id

    ).order_by(

                   PredictionHistory.created_at.desc()

    ).all()


    return render_template(

        'history.html',

        predictions=predictions

    )
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_disease():

    if request.method == "POST":

        age = float(request.form.get("age"))
        sex = float(request.form.get("sex"))
        cp = float(request.form.get("cp"))
        trestbps = float(request.form.get("trestbps"))
        chol = float(request.form.get("chol"))
        fbs = float(request.form.get("fbs"))
        restecg = float(request.form.get("restecg"))
        thalach = float(request.form.get("thalach"))
        exang = float(request.form.get("exang"))
        oldpeak = float(request.form.get("oldpeak"))
        slope = float(request.form.get("slope"))
        ca = float(request.form.get("ca"))
        thal = float(request.form.get("thal"))

        # Create feature array
        features = np.array([
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        prediction = best_model.predict(features_scaled)

        probability = best_model.predict_proba(features_scaled)

        risk_percentage = probability[0][1] * 100

        if prediction[0] == 1:
                result = "Heart Disease Detected"
        else:
                result = "No Heart Disease Detected"
        if risk_percentage < 30:
              risk_level = "Low Risk"
        elif risk_percentage < 70:
              risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
        new_prediction = PredictionHistory(

              user_id=current_user.id,

              risk_percentage=risk_percentage,

              risk_level=risk_level,

              result=result
)

        db.session.add(new_prediction)

        db.session.commit()
        # Return result
        return render_template(
            "home.html",
             prediction=result,
             risk_percentage=risk_percentage,
             risk_level=risk_level
        )


    # If GET request
    return render_template("home.html")


if __name__ == '__main__':
    app.run(debug=True)