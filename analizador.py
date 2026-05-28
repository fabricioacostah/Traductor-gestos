import cv2
import mediapipe as mp

# =======================================================================
# IMPORTACIONES EXPLÍCITAS PARA PYINSTALLER
# Esto obliga a GitHub y PyInstaller a incluir MediaPipe dentro del .exe
# =======================================================================
import mediapipe.solutions.hands as mp_hands
import mediapipe.solutions.drawing_utils as mp_drawing
import mediapipe.solutions.drawing_styles as mp_drawing_styles

def main():
    # 1. Intentar abrir la cámara web (0 es usualmente la cámara integrada)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    # 2. Configurar el modelo de Inteligencia Artificial de MediaPipe
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:
        
        print("=========================================")
        print("  Analizador de Gestos Activo")
        print("  Presiona la tecla 'q' en Windows para salir")
        print("=========================================")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Esperando la señal de la cámara...")
                continue

            # Efecto espejo para que el movimiento sea natural (derecha es derecha)
            image = cv2.flip(image, 1)

            # Convertir el color: OpenCV usa BGR y MediaPipe necesita RGB
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen con la IA para buscar manos
            results = hands.process(image_rgb)

            # Dibujar los resultados en la imagen original
            image.flags.writeable = True
            
            # Si el algoritmo encuentra manos en la pantalla...
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar los puntos y las líneas que unen los dedos
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                # Poner un texto en pantalla si detecta manos
                cv2.putText(image, "Mano Detectada", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Poner texto si la pantalla está vacía
                cv2.putText(image, "Buscando manos...", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 3. Mostrar la ventana con el video en tiempo real
            cv2.imshow('Traductor y Analizador de Gestos', image)
            
            # Cerrar la ventana si el usuario presiona la letra 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # 4. Limpieza de recursos al cerrar
    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")

if __name__ == '__main__':
    main()
