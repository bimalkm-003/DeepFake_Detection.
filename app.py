import os

from flask import Flask, render_template, url_for, request, session, flash, redirect, send_from_directory
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "Rounak2004"
app.permanent_session_lifetime = timedelta(minutes=10)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    password = db.Column(db.String(200), nullable=False)


class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=db.func.now())


def allowed_file(filename: str) -> bool:
    allowed_extensions = {
        "mp4", "mov", "avi", "mkv",
        "png", "jpg", "jpeg", "gif", "bmp", "webp",
        "mp3", "wav", "ogg", "m4a", "flac", "aac",
    }
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/")
def home():
    if "user" in session:
        return render_template("user.html", user=session["user"])
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return redirect(url_for("register"))

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registered in successfully", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier']
        password = request.form['password']

        if '@' in identifier:
            user = User.query.filter_by(email=identifier).first()
        else:
            user = User.query.filter_by(username=identifier).first()

        if user and check_password_hash(user.password, password):
            session['user'] = user.username
            session['email'] = user.email
            return redirect(url_for('user'))
        else:
            flash('Invalid credentials', "error")

    return render_template('login.html')



@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    message = None

    if request.method == "POST":
        session["display_name"] = request.form.get("display_name", "").strip()
        session["bio"]          = request.form.get("bio", "").strip()
        session["email"]        = request.form.get("email", "").strip()
        message = "Profile updated successfully."

    return render_template(
        "profile.html",
        user         = session.get("user"),
        display_name = session.get("display_name", session.get("user", "")),
        bio          = session.get("bio", ""),
        email        = session.get("email", ""),
        message      = message,
    )


@app.route("/user")
def user():
    if "user" in session:
        return render_template("user.html", user=session["user"])
    else:
        flash("You are not logged in", "error")
        return redirect(url_for("login"))





@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        media = request.files.get("media")

        if not media or media.filename == "":
            flash("Please choose an image, audio, or video file to upload.", "error")
            return redirect(url_for("upload"))
        elif not allowed_file(media.filename):
            flash("Unsupported file type. Upload an image, audio, or video file.", "error")
            return redirect(url_for("upload"))

        filename = secure_filename(media.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        media.save(filepath)

        new_file = File(filename=filename)
        db.session.add(new_file)
        db.session.commit()

        flash("File uploaded successfully.", "success")
        return redirect(url_for("upload", uploaded=filename))

    media_url = None
    media_type = None
    uploaded = request.args.get("uploaded")

    if uploaded:
        media_url = url_for("uploaded_file", filename=uploaded)
        ext = uploaded.rsplit(".", 1)[1].lower()
        if ext in {"png", "jpg", "jpeg", "gif", "bmp", "webp"}:
            media_type = "image"
        elif ext in {"mp3", "wav", "ogg", "m4a", "flac", "aac"}:
            media_type = "audio"
        else:
            media_type = "video"

    return render_template("upload.html", media_url=media_url, media_type=media_type, uploaded_filename=uploaded)

@app.route('/delete-file', methods=['POST'])
def delete_file():
    filename = request.form.get('filename')

    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            flash('File deleted successfully.')

    return redirect(url_for('upload'))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        filename = request.form.get('filename')

        if not filename:
            return redirect(url_for('upload'))

        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        if not os.path.exists(file_path):
            return redirect(url_for('upload'))

        # ---------------------------------
        # YOUR DETECTION MODEL GOES HERE
        # ---------------------------------
        return render_template(
            'detect.html',
            filename=filename,
            file_path=file_path
        )

    return render_template('detect.html')
    

    


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

    
