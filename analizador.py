import cv2
import mediapipe as mp

# 1. Inicializar MediaPipe configurado para DOS MANOS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# IDs de las puntas de los dedos
tip_ids = [4, 8, 12, 16, 20]

# Diccionario de comunicación
vocabulario_bimanual = {
    (5, 5): "¡HOLA! / ¡BIENVENIDOS!",
    (0, 0): "GRACIAS (Muestras Respeto)",
    (5, 0): "POR FAVOR...",
    (0, 5): "LO SIENTO / DISCULPA",
    (2, 2): "PAZ Y AMOR / AMISTAD",
    (1, 1): "SÍ / DE ACUERDO",
    (1, 5): "NECESITO AYUDA / SOCORRO",
    (5, 2): "¡FELICITACIONES! / BUEN TRABAJO"
}

vocabulario_unamanual = {
    5: "¡ADIÓS! / HASTA LUEGO",
    2: "TODO BIEN / OK",
    0: "ESPERA / PAUSA"
}

cap = cv2.VideoCapture(0)
print("Intérprete bimanual con interfaz ampliada iniciado.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape

    # --- PANEL VISUAL DEL DICCIONARIO (MÁS GRANDE Y LEGIBLE) ---
    # Ampliamos la caja a 480px de ancho y 320px de alto
    cv2.rectangle(frame, (15, 15), (490, 335), (35, 25, 15), -1)  # Fondo oscuro sólido
    cv2.rectangle(frame, (15, 15), (490, 335), (255, 128, 0), 3)  # Borde naranja más grueso (grosor 3)
    
    # Título principal con tipografía grande
    cv2.putText(frame, "DICCIONARIO GESTUAL", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 128, 0), 2, cv2.LINE_AA)
    
    # Lista de comandos con escala aumentada a 0.55 e interlineado de 32px para evitar amontonamientos
    cv2.putText(frame, "5 + 5 ded. -> ¡HOLA! / BIENVENIDOS", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "0 + 0 ded. -> GRACIAS", (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "5 + 0 ded. -> POR FAVOR", (30, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "0 + 5 ded. -> LO SIENTO / DISCULPA", (30, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "2 + 2 ded. -> PAZ Y AMOR", (30, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "1 + 1 ded. -> SÍ / DE ACUERDO", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "1 + 5 ded. -> NECESITO AYUDA", (30, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Recordatorio de una sola mano destacado en color cian
    cv2.putText(frame, "Solo 1 mano abierta (5) -> ¡ADIÓS!", (30, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1, cv2.LINE_AA)

    # Procesamiento de imágenes
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    dedos_izq = None
    dedos_der = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lado_anatomico = handedness.classification[0].label 
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dedos_abiertos = []

            # Lógica del Pulgar
            if lado_anatomico == "Left":
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    dedos_abiertos.append(1)
                else: dedos_abiertos.append(0)
            else: 
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    dedos_abiertos.append(1)
                else: dedos_abiertos.append(0)

            # Lógica de los otros 4 dedos
            for id in range(1, 5):
                if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                    dedos_abiertos.append(1)
                else:
                    dedos_abiertos.append(0)

            total_dedos_mano = dedos_abiertos.count(1)

            if lado_anatomico == "Left":
                dedos_izq = total_dedos_mano
            else:
                dedos_der = total_dedos_mano

    # --- DETERMINAR TRADUCCIÓN ---
    color_banner = (40, 40, 40)
    
    if dedos_izq is not None and dedos_der is not None:
        mensaje_pantalla = vocabulario_bimanual.get((dedos_izq, dedos_der), f"Combinación ({dedos_izq} + {dedos_der})")
        color_texto = (0, 255, 0) 
        color_banner = (15, 45, 15)
        status_manos = f"Mano Izq: {dedos_izq} | Mano Der: {dedos_der}"
    
    elif dedos_izq is not None or dedos_der is not None:
        dedos_activos = dedos_izq if dedos_izq is not None else dedos_der
        mano_activa = "Izquierda" if dedos_izq is not None else "Derecha"
        mensaje_pantalla = vocabulario_unamanual.get(dedos_activos, "Formando palabra...")
        color_texto = (0, 255, 255) 
        status_manos = f"Mano {mano_activa} detectada ({dedos_activos} dedos)"
    
    else:
        mensaje_pantalla = "SISTEMA EN ESPERA..."
        color_texto = (0, 0, 255) 
        status_manos = "Escaneando entorno visual..."

    # --- INTERFAZ INFERIOR (TEXTO GRANDE) ---
    cv2.rectangle(frame, (0, alto - 90), (ancho, alto), color_banner, -1)
    
    # Telemetría de las manos
    cv2.putText(frame, status_manos, (25, alto - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Texto traducido principal (Tamaño 0.9 y Grosor 2 para máxima legibilidad)
    cv2.putText(frame, f"TRADUCCION: {mensaje_pantalla}", (25, alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_texto, 2, cv2.LINE_AA)

    cv2.imshow("Traductor Inteligente de Expresiones Gestuales", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()