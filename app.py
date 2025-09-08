from flask import Flask, render_template, request, redirect, url_for, send_file
import sqlite3
import cv2
import time
import os
from datetime import datetime
import face_recognition
import numpy as np
import csv

app = Flask(__name__)
UPLOAD_FOLDER = "static/students"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database setup
def init_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT,
                    course TEXT,
                    lecturer TEXT,
                    delegate TEXT,
                    time TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS students
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Home page
@app.route('/')
def index():
    return render_template("index.html")

# Delegate page
@app.route('/delegate', methods=["GET", "POST"])
def delegate():
    if request.method == "POST":
        course = request.form["course"]
        lecturer = request.form["lecturer"]
        delegate = request.form["delegate"]
        return redirect(url_for("attendance", course=course, lecturer=lecturer, delegate=delegate))
    return render_template("delegate.html")

# Register students page
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        file = request.files["photo"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO students (name, image_path) VALUES (?,?)", (name, path))
            conn.commit()
            conn.close()

    return render_template("register.html")

# Attendance page with face recognition
@app.route('/attendance')
def attendance():
    course = request.args.get("course")
    lecturer = request.args.get("lecturer")
    delegate = request.args.get("delegate")

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, image_path FROM students")
    students = cursor.fetchall()
    conn.close()

    known_encodings = []
    known_names = []
    for name, path in students:
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(name)

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    marked_students = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)

        for encoding, face in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"
            if True in matches:
                index = matches.index(True)
                name = known_names[index]

                if name not in marked_students:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn = sqlite3.connect("attendance.db")
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO attendance (student_name, course, lecturer, delegate, time) VALUES (?,?,?,?,?)",
                                   (name, course, lecturer, delegate, now))
                    conn.commit()
                    conn.close()
                    marked_students.add(name)

            top, right, bottom, left = face
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Attendance System", frame)

        if time.time() - start_time > 900:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Attendance session ended after 15 minutes."

# Reports page
@app.route('/report')
def report():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    conn.close()
    return render_template("report.html", records=records)

# Export attendance to CSV
@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    conn.close()

    filename = "attendance_report.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Student", "Course", "Lecturer", "Delegate", "Time"])
        writer.writerows(records)
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
