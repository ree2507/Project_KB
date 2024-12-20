import cv2
import os

# Path utama dataset
dataset_path = "dataset/wajah/"

# Fungsi untuk mengambil gambar wajah
def capture_faces():
    # Meminta input nama folder
    name = input("Masukkan nama untuk folder dataset wajah: ").strip()
    folder_path = os.path.join(dataset_path, name)

    # Buat folder jika belum ada
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{name}' berhasil dibuat di {folder_path}")
    else:
        print(f"Folder '{name}' sudah ada, gambar akan ditambahkan ke folder ini.")

    # Load Haarcascade untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Buka kamera
    cap = cv2.VideoCapture(0)
    print("Tekan 'q' untuk berhenti mengambil gambar.")

    count = 1  # Counter untuk penamaan gambar
    # Cek jika sudah ada file dalam folder, lanjutkan penomoran
    existing_files = os.listdir(folder_path)
    if existing_files:
        count += len(existing_files)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat mengakses kamera!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Crop area wajah
            file_name = os.path.join(folder_path, f"{name}{count}.jpg")
            cv2.imwrite(file_name, face)  # Simpan gambar wajah
            print(f"Gambar {file_name} berhasil disimpan!")
            count += 1

            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Proses pengambilan wajah selesai.")

# Jalankan fungsi pengambilan wajah
if __name__ == "__main__":
    capture_faces()
