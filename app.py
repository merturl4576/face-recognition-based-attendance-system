import os
import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template
from sklearn.neighbors import KNeighborsClassifier

# Flask uygulaması tanımlanıyor
app = Flask(__name__)
N_IMAGES = 10

# Bugünün tarihi iki farklı formatta kaydediliyor
DATE_TODAY = date.today().strftime("%m_%d_%y")
DATE_TODAY_DISPLAY = date.today().strftime("%d-%B-%Y")

# Klasörler kontrol ediliyor
os.makedirs("Attendance", exist_ok=True)
os.makedirs("static/faces", exist_ok=True)

ATTENDANCE_FILE = f"Attendance/Attendance-{DATE_TODAY}.csv"
if not os.path.isfile(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
        f.write("Name,Roll,Time")

# Haar Cascade tanımlayıcısı yükleniyor
CASCADE_PATH = "C:/ECHO/MINIPROJECT/facerecpro/face-recognition-based-attendance-system/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def total_registered_users():
    return len(os.listdir("static/faces"))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return faces
    except Exception:
        return []


def identify_face(face_array):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(face_array)


def train_model():
    faces = []
    labels = []
    for user in os.listdir("static/faces"):
        user_path = os.path.join("static/faces", user)
        for imgname in os.listdir(user_path):
            img = cv2.imread(os.path.join(user_path, imgname))
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")


def extract_attendance():
    df = pd.read_csv(ATTENDANCE_FILE)
    return df["Name"], df["Roll"], df["Time"], len(df)


def add_attendance(name):
    username, userid = name.split("_")
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(ATTENDANCE_FILE)
    if int(userid) not in df["Roll"].values:
        with open(ATTENDANCE_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{username},{userid},{current_time}")


def get_all_users():
    names, rolls = [], []
    userlist = os.listdir("static/faces")
    for user in userlist:
        name, roll = user.split("_")
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)


def delete_user_folder(user_path):
    for img in os.listdir(user_path):
        os.remove(os.path.join(user_path, img))
    os.rmdir(user_path)


@app.route("/")
def home():
    names, rolls, times, count = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=count,
        totalreg=total_registered_users(),
        datetoday2=DATE_TODAY_DISPLAY,
    )


@app.route("/listusers")
def list_users():
    userlist, names, rolls, count = get_all_users()
    return render_template(
        "listusers.html",
        userlist=userlist,
        names=names,
        rolls=rolls,
        l=count,
        totalreg=total_registered_users(),
        datetoday2=DATE_TODAY_DISPLAY,
    )


@app.route("/deleteuser", methods=["GET"])
def delete_user():
    user = request.args.get("user")
    user_path = os.path.join("static/faces", user)
    delete_user_folder(user_path)

    model_path = "static/face_recognition_model.pkl"
    if not os.listdir("static/faces") and os.path.isfile(model_path):
        os.remove(model_path)

    try:
        train_model()
    except Exception:
        pass

    userlist, names, rolls, count = get_all_users()
    return render_template(
        "listusers.html",
        userlist=userlist,
        names=names,
        rolls=rolls,
        l=count,
        totalreg=total_registered_users(),
        datetoday2=DATE_TODAY_DISPLAY,
    )


@app.route("/start", methods=["GET"])
def start():
    names, rolls, times, count = extract_attendance()
    model_path = "static/face_recognition_model.pkl"
    if not os.path.isfile(model_path):
        return render_template(
            "home.html",
            names=names,
            rolls=rolls,
            times=times,
            l=count,
            totalreg=total_registered_users(),
            datetoday2=DATE_TODAY_DISPLAY,
            mess="Trained model bulunamadı. Lütfen yeni bir kullanıcı ekleyin.",
        )

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if faces:
            x, y, w, h = faces[0]
            face_img = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            person = identify_face(face_img.reshape(1, -1))[0]
            add_attendance(person)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            cv2.putText(frame, person, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, count = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=count,
        totalreg=total_registered_users(),
        datetoday2=DATE_TODAY_DISPLAY,
    )


@app.route("/add", methods=["POST"])
def add_user():
    newusername = request.form["newusername"]
    newuserid = request.form["newuserid"]
    user_folder = os.path.join("static/faces", f"{newusername}_{newuserid}")

    os.makedirs(user_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    i = j = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f"Images Captured: {i}/{N_IMAGES}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

            if j % 5 == 0:
                face_img = frame[y:y + h, x:x + w]
                img_name = f"{newusername}_{i}.jpg"
                cv2.imwrite(os.path.join(user_folder, img_name), face_img)
                i += 1
            j += 1

        if j >= N_IMAGES * 5 or cv2.waitKey(1) == 27:
            break

        cv2.imshow("Adding new User", frame)

    cap.release()
    cv2.destroyAllWindows()

    train_model()

    names, rolls, times, count = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=count,
        totalreg=total_registered_users(),
        datetoday2=DATE_TODAY_DISPLAY,
    )


if __name__ == "__main__":
    app.run(debug=True)
