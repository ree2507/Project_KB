import cv2
import face_recognition
import os
import numpy as np
from sklearn.svm import SVC
import joblib  # Untuk menyimpan dan memuat model SVM

# Path dataset wajah
dataset_path = "dataset/wajah/"
model_path = "svm_face_model.pkl"  # Path untuk menyimpan model SVM

# Fungsi untuk memuat wajah dan nama dari dataset
def load_face_encodings():
    face_encodings = []
    face_names = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)

            # Encode wajah
            face_encoding = face_recognition.face_encodings(image)
            if face_encoding:
                face_encodings.append(face_encoding[0])
                face_names.append(person_name)

    return face_encodings, face_names

# Fungsi untuk melatih model SVM
def train_svm_model(encodings, names):
    print("Melatih model SVM...")
    model = SVC(kernel="linear", probability=True)  # SVM dengan kernel linear
    model.fit(encodings, names)
    joblib.dump(model, model_path)  # Simpan model ke file
    print("Model SVM berhasil dilatih dan disimpan!")

# Fungsi untuk mengenali wajah menggunakan SVM
def recognize_faces_svm(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Unknown"

        # Prediksi kelas menggunakan model SVM
        probabilities = model.predict_proba([face_encoding])[0]
        best_match_index = np.argmax(probabilities)
        if probabilities[best_match_index] > 0.6:  # Confidence threshold
            name = model.classes_[best_match_index]

        # Gambar kotak dan nama di sekitar wajah
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Load dataset dan model SVM
if not os.path.exists(model_path):
    print("Memuat dataset wajah dan melatih model...")
    known_encodings, known_names = load_face_encodings()
    train_svm_model(known_encodings, known_names)
else:
    print("Memuat model SVM yang sudah ada...")
    model = joblib.load(model_path)

# Buka kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Kenali wajah menggunakan SVM
    recognize_faces_svm(frame, model)

    # Tampilkan frame
    cv2.imshow("Face Recognition with SVM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
