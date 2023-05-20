import cv2
import mediapipe as mp
import numpy as np

# Inicializar la detección de la cara y la malla facial de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Valor de la tecla ESC para salir
tecla_salida = 27

def procesar_malla_facial(frame, landmarks):
    # Crear la malla utilizando los puntos faciales
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        cv2.circle(mask, (x, y), 1, (255), -1)

    # Aplicar filtros de maquillaje
    # ...

    return frame

while True:
    # Leer el fotograma desde la cámara
    ret, frame_captureado = cap.read()

    if not ret:
        print("Error al capturar fotograma.")
        break

    # Procesar el fotograma a través del modelo de Face Mesh
    results = face_mesh.process(cv2.cvtColor(frame_captureado, cv2.COLOR_BGR2RGB))

    # Obtener las coordenadas de los 340 puntos faciales
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark[:340]

        # Crear la malla utilizando los puntos faciales
        frame_captureado = procesar_malla_facial(frame_captureado, face_landmarks)

    # Mostrar el fotograma procesado
    cv2.imshow("Face Mesh", frame_captureado)

    # Esperar por la tecla ESC para salir
    tecla = cv2.waitKey(1)
    if tecla == tecla_salida:
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()









