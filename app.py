"""Face Recognition Attendance System with Flask and OpenCV"""

import os
from datetime import date, datetime

import cv2  # pylint: disable=no-member
import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
NIMGS = 10  # Number of images per user for training

# Set up current date
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load face detector
face_detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)  # pylint: disable=no-member

# Ensure necessary directories exist
for folder in ['Attendance', 'static', 'static/faces']:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# Create today's attendance file if not exists
ATTENDANCE_FILE = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as attend_file:
        attend_file.write('Name,Roll,Time')


def totalreg():
    """Returns the total number of registered people"""
    return len(os.listdir('static/faces'))


def extract_faces(img):
    """Extract faces from an image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
    faces = face_detector.detectMultiScale(
        gray, 1.2, 5, minSize=(20, 20)
    )
    return faces


def identify_face(face_array):
    """Identify face using trained model"""
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(face_array)


def train_model():
    """Train the KNN model with saved face data"""
    faces = []
    labels = []
    for user_dir in os.listdir('static/faces'):
        for img_file in os.listdir(f'static/faces/{user_dir}'):
            img_path = f'static/faces/{user_dir}/{img_file}'
            img = cv2.imread(img_path)  # pylint: disable=no-member
            resized = cv2.resize(img, (50, 50))  # pylint: disable=no-member
            faces.append(resized.flatten())
            labels.append(user_dir)
    if faces and labels:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(faces, labels)
        joblib.dump(model, 'static/face_recognition_model.pkl')


def extract_attendance():
    """Extract attendance from today's file"""
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    df = pd.read_csv(ATTENDANCE_FILE)
    return df.values.tolist()


@app.route('/')
def home():
    """Render homepage"""
    names = os.listdir('static/faces')
    return render_template(
        'home.html',
        names=names,
        totalreg=totalreg(),
        datetoday2=datetoday2
    )


@app.route('/start', methods=['GET'])
def start():
    """Start webcam and perform face recognition"""
    cap = cv2.VideoCapture(0)  # pylint: disable=no-member
    if not cap.isOpened():
        return "Camera couldn't be accessed"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (50, 50))  # pylint: disable=no-member
            identified = identify_face([resized.flatten()])
            name = identified[0]
            current_time = datetime.now().strftime("%H:%M:%S")
            df = pd.read_csv(ATTENDANCE_FILE)
            if name not in df['Name'].values:
                with open(ATTENDANCE_FILE, 'a', encoding='utf-8') as file_out:
                    file_out.write(f'\n{name},{name},{current_time}')
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )  # pylint: disable=no-member
            cv2.putText(
                frame, name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )  # pylint: disable=no-member
        cv2.imshow("Attendance", frame)  # pylint: disable=no-member
        if cv2.waitKey(1) & 0xFF == ord('q'):  # pylint: disable=no-member
            break

    cap.release()
    cv2.destroyAllWindows()  # pylint: disable=no-member
    return render_template(
        'home.html',
        names=os.listdir('static/faces'),
        totalreg=totalreg(),
        datetoday2=datetoday2,
        mess='Attendance taken successfully'
    )


@app.route('/add', methods=['POST'])
def add():
    """Add a new face to the system"""
    newuser = request.form['newuser']
    if not newuser:
        return render_template(
            'home.html',
            names=os.listdir('static/faces'),
            totalreg=totalreg(),
            datetoday2=datetoday2,
            mess='Please enter a name'
        )

    user_dir = f'static/faces/{newuser}'
    if not os.path.isdir(user_dir):
        os.makedirs(user_dir)

    cap = cv2.VideoCapture(0)  # pylint: disable=no-member
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (50, 50))  # pylint: disable=no-member
            cv2.imwrite(
                f'{user_dir}/{count}.jpg', resized
            )  # pylint: disable=no-member
        cv2.imshow("Adding new user", frame)  # pylint: disable=no-member
        if (
            cv2.waitKey(1) & 0xFF == ord('q')
            or count == NIMGS
        ):  # pylint: disable=no-member
            break

    cap.release()
    cv2.destroyAllWindows()  # pylint: disable=no-member
    train_model()
    return render_template(
        'home.html',
        names=os.listdir('static/faces'),
        totalreg=totalreg(),
        datetoday2=datetoday2,
        mess='New user added successfully'
    )


@app.route('/attendance')
def attendance():
    """Display attendance records"""
    records = extract_attendance()
    return render_template(
        'attendance.html',
        records=records,
        totalreg=totalreg(),
        datetoday2=datetoday2
    )


@app.route('/delete', methods=['POST'])
def delete():
    """Delete a user"""
    username = request.form['username']
    user_path = f'static/faces/{username}'
    if os.path.isdir(user_path):
        for file in os.listdir(user_path):
            os.remove(os.path.join(user_path, file))
        os.rmdir(user_path)
        train_model()
    return render_template(
        'home.html',
        names=os.listdir('static/faces'),
        totalreg=totalreg(),
        datetoday2=datetoday2,
        mess='User deleted successfully'
    )


if __name__ == '__main__':
    app.run(debug=True)
