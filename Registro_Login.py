import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import threading
import csv

# Configuración de la carpeta para almacenar las imágenes
TRAINING_IMAGES_FOLDER = os.path.join(os.getcwd(), "training_images")
IMG_SIZE = (128, 128)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear la carpeta si no existe
if not os.path.exists(TRAINING_IMAGES_FOLDER):
    os.makedirs(TRAINING_IMAGES_FOLDER)

# Archivo CSV para almacenar los datos de los usuarios
USERS_CSV = os.path.join(os.getcwd(), "usuarios.csv")

# Crear el archivo CSV si no existe
if not os.path.exists(USERS_CSV):
    with open(USERS_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["usuario", "correo", "imagen"])

# Funciones de procesamiento de imágenes
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    return img_resized

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def load_dataset():
    images, labels = [], []
    for filename in os.listdir(TRAINING_IMAGES_FOLDER):
        if filename.endswith(".jpg"):
            username = filename.split(".")[0]
            img_path = os.path.join(TRAINING_IMAGES_FOLDER, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_processed = preprocess_image(img)
            features = extract_hog_features(img_processed)
            images.append(features)
            labels.append(username)
    print(f"Dataset cargado: {len(images)} imágenes")  # Depuración
    return np.array(images), np.array(labels)

def train_model():
    X, y = load_dataset()
    if X.size == 0:
        raise ValueError("No se encontraron imágenes en la carpeta 'training_images'.")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)
    print("Modelo entrenado con éxito.")  # Mensaje de depuración
    return model, label_encoder

def authenticate_user(model, label_encoder, image):
    input_img = preprocess_image(image)
    input_features = extract_hog_features(input_img).reshape(1, -1)
    prediction = model.predict(input_features)
    predicted_label = label_encoder.inverse_transform(prediction)
    probabilities = model.predict_proba(input_features)
    confidence = np.max(probabilities)
    print(f"Predicción: {predicted_label}, Confianza: {confidence}")  # Mensaje de depuración
    threshold = 0.2  # Reducido para mejorar la detección
    if confidence >= threshold:
        return predicted_label[0]
    return None

def capture_images(username, correo, mode, num_images=5):
    def run_capture():
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("No se pudo acceder a la cámara.")  # Depuración
                messagebox.showerror("Error", "No se pudo acceder a la cámara.")
                return
            face_detected = False
            images_captured = 0
            while images_captured < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo capturar la imagen.")  # Depuración
                    messagebox.showerror("Error", "No se pudo capturar la imagen.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_detected = True

                cv2.imshow("Captura de Imagen - Presiona 's' para guardar", frame)
                key = cv2.waitKey(1)
                if key == ord('s') and face_detected:
                    face = frame[y:y+h, x:x+w]
                    img_resized = cv2.resize(face, IMG_SIZE)
                    if mode == "register":
                        img_path = os.path.join(TRAINING_IMAGES_FOLDER, f"{username}_{images_captured}.jpg")
                        print(f"Guardando imagen en {img_path}")  # Depuración: Verificar la ruta de guardado
                        cv2.imwrite(img_path, img_resized)  # Guardar solo la región de la cara redimensionada
                        if os.path.exists(img_path):
                            print("Imagen guardada exitosamente en la carpeta correcta.")  # Depuración
                            with open(USERS_CSV, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([username, correo, img_path])
                        else:
                            print("Error al guardar la imagen en la carpeta correcta.")  # Depuración
                        images_captured += 1
                        print(f"Imagen {images_captured} capturada.")  # Depuración
                    elif mode == "login":
                        username_recognized = authenticate_user(model, label_encoder, img_resized)
                        if username_recognized:
                            print(f"Usuario reconocido: {username_recognized}")  # Depuración
                            messagebox.showinfo("Autenticación Exitosa", f"Bienvenido, {username_recognized}!")
                        else:
                            print("Usuario no reconocido.")  # Depuración
                            messagebox.showerror("Autenticación Fallida", "Usuario no reconocido.")
                        break
                elif key == 27:  # Esc key to exit
                    print("Cerrando la captura de imagen.")  # Depuración
                    break
        except Exception as e:
            print(f"Error en la captura de imagen: {str(e)}")  # Depuración
            messagebox.showerror("Error", str(e))
        finally:
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()

        if mode == "register":
            messagebox.showinfo("Registro Completo", "Imágenes capturadas y guardadas exitosamente.")
            update_model()  # Actualizar el modelo después de guardar las imágenes

    capture_thread = threading.Thread(target=run_capture)
    capture_thread.start()

def iniciar_sesion():
    login_window = tk.Toplevel()
    login_window.title("Iniciar Sesión")
    login_window.geometry("300x350")
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TLabel", font=("Helvetica", 14))
    frame = ttk.Frame(login_window, padding="20 20 20 20")
    frame.pack(fill=tk.BOTH, expand=True)
    titulo = ttk.Label(frame, text="Iniciar Sesión", font=("Helvetica", 18, "bold"))
    titulo.pack(pady=10)
    etiqueta_usuario = ttk.Label(frame, text="Usuario:")
    etiqueta_usuario.pack(pady=5)
    entrada_usuario = ttk.Entry(frame)
    entrada_usuario.pack(pady=5, fill=tk.X)
    etiqueta_contraseña = ttk.Label(frame, text="Contraseña:")
    etiqueta_contraseña.pack(pady=5)
    entrada_contraseña = ttk.Entry(frame, show="*")
    entrada_contraseña.pack(pady=5, fill=tk.X)

    def verificar_credenciales():
        usuario = entrada_usuario.get()
        contraseña = entrada_contraseña.get()
        # Aquí puedes agregar la lógica para verificar las credenciales
        with open(USERS_CSV, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la cabecera
            for row in reader:
                if row[0] == usuario and row[2] == contraseña:
                    messagebox.showinfo("Inicio de sesión", "Inicio de sesión exitoso")
                    return
        messagebox.showerror("Inicio de sesión", "Usuario o contraseña incorrectos")

    btn_enviar = ttk.Button(frame, text="Iniciar sesión", command=verificar_credenciales)
    btn_enviar.pack(pady=5, fill=tk.X)
    btn_reconocimiento_facial = ttk.Button(frame, text="Inicio Facial", command=lambda: capture_images(entrada_usuario.get(), None, "login", num_images=1))
    btn_reconocimiento_facial.pack(pady=5, fill=tk.X)

def registrarse():
    register_window = tk.Toplevel()
    register_window.title("Registrarse")
    register_window.geometry("300x350")
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TLabel", font=("Helvetica", 14))
    frame = ttk.Frame(register_window, padding="20 20 20 20")
    frame.pack(fill=tk.BOTH, expand=True)
    titulo = ttk.Label(frame, text="Registrarse", font=("Helvetica", 18, "bold"))
    titulo.pack(pady=10)
    etiqueta_usuario = ttk.Label(frame, text="Usuario:")
    etiqueta_usuario.pack(pady=5)
    entrada_usuario = ttk.Entry(frame)
    entrada_usuario.pack(pady=5, fill=tk.X)
    etiqueta_correo = ttk.Label(frame, text="Correo:")
    etiqueta_correo.pack(pady=5)
    entrada_correo = ttk.Entry(frame)
    entrada_correo.pack(pady=5, fill=tk.X)
    etiqueta_contraseña = ttk.Label(frame, text="Contraseña:")
    etiqueta_contraseña.pack(pady=5)
    entrada_contraseña = ttk.Entry(frame, show="*")
    entrada_contraseña.pack(pady=5, fill=tk.X)

    def registrar_usuario():
        usuario = entrada_usuario.get()
        correo = entrada_correo.get()
        print(f"Iniciando captura para registrar usuario: {usuario}")  # Depuración
        capture_images(usuario, correo, "register", num_images=5)

    btn_registrar = ttk.Button(frame, text="Registrar", command=registrar_usuario)
    btn_registrar.pack(pady=5, fill=tk.X)
    btn_registro_facial = ttk.Button(frame, text="Registro Facial", command=lambda: capture_images(entrada_usuario.get(), entrada_correo.get(), "register", num_images=5))
    btn_registro_facial.pack(pady=5, fill=tk.X)

def update_model():
    global model, label_encoder
    try:
        model, label_encoder = train_model()
        messagebox.showinfo("Modelo Actualizado", "El modelo ha sido actualizado con nuevos datos.")
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Entrenar el modelo (esto se hace una vez al inicio)
try:
    model, label_encoder = train_model()
except ValueError as e:
    messagebox.showerror("Error", str(e))

ventana = tk.Tk()
ventana.title("Proyecto Reconocimiento Facial")
ventana.geometry("400x300")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=10)
style.configure("TLabel", font=("Helvetica", 14))

frame = ttk.Frame(ventana, padding="20 20 20 20")
frame.pack(fill=tk.BOTH, expand=True)

titulo = ttk.Label(frame, text="Sistema de Registro y Login", font=("Helvetica", 18, "bold"))
titulo.pack(pady=20)

btn_registrarse = ttk.Button(frame, text="Registrarse", command=registrarse)
btn_registrarse.pack(pady=10, fill=tk.X)

btn_iniciar_sesion = ttk.Button(frame, text="Iniciar Sesión", command=iniciar_sesion)
btn_iniciar_sesion.pack(pady=10, fill=tk.X)

ventana.mainloop()
