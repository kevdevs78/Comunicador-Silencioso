import cv2
import os
import numpy as np
from src.hand_detector import HandDetector
from src.gif_generator import GIFGenerator

class DataCollector:
    def __init__(self):
        self.data_path = "data/raw"
        self.gif_path = "data/gifs"
        self.cap = cv2.VideoCapture(0)
        self.hand_detector = HandDetector(max_num_hands=2)  # ✅ Detectar hasta 2 manos
        self.gif_generator = GIFGenerator()

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.gif_path, exist_ok=True)

    def collect_data(self, sign_name, num_samples):
        print(f"📸 Capturando {num_samples} muestras para la seña: {sign_name}")
        print("Muestra tus dos manos frente a la cámara y presiona 'c' para capturar.")
        print("Presiona 'q' para salir antes de completar.")

        frames_for_gif = []
        sample_count = 0

        while sample_count < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # 🔹 Detectar manos (hasta 2)
            hand_landmarks_list, handedness_list, _ = self.hand_detector.detect_hands(frame)

            if hand_landmarks_list:
                all_features = []

                # 🔹 Procesar cada mano detectada (una o dos)
                for i, hand_landmarks in enumerate(hand_landmarks_list):
                    handedness = handedness_list[i]

                    # Dibujar landmarks en pantalla
                    self.hand_detector.draw_landmarks(display_frame, hand_landmarks)

                    # Extraer características de cada mano
                    landmarks_features = self.hand_detector.extract_landmarks_features(hand_landmarks, frame.shape)
                    angle_features = self.hand_detector.get_hand_angle_features(hand_landmarks)
                    hand_features = np.concatenate([landmarks_features, angle_features])
                    all_features.extend(hand_features)

                    # Dibujar rectángulo ROI
                    x_min, y_min, x_max, y_max = self.hand_detector.get_hand_roi(frame, hand_landmarks, padding=30)
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Mano {i+1}: {handedness.classification[0].label}",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 🔹 Si solo hay 1 mano, agregar ceros para mantener tamaño fijo
                if len(hand_landmarks_list) == 1:
                    all_features.extend([0] * len(hand_features))  # relleno con ceros

                cv2.putText(display_frame, f"Muestra: {sample_count}/{num_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Presiona 'c' para capturar",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Captura de Señas - Entrenamiento (2 Manos)", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    try:
                        # Guardar características combinadas
                        features_path = os.path.join(self.data_path, f"{sign_name}_{sample_count}.npy")
                        np.save(features_path, np.array(all_features))

                        # Guardar imagen de referencia
                        img_path = os.path.join(self.data_path, f"{sign_name}_{sample_count}.jpg")
                        cv2.imwrite(img_path, display_frame)

                        # Agregar al GIF
                        gif_frame = cv2.resize(display_frame, (320, 240))
                        frames_for_gif.append(gif_frame)

                        print(f"✅ Muestra {sample_count} capturada ({len(all_features)} características)")
                        sample_count += 1

                    except Exception as e:
                        print(f"⚠️ Error capturando muestra: {e}")

                elif key == ord('q'):
                    break
            else:
                cv2.putText(display_frame, "No se detectan manos", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Captura de Señas - Entrenamiento (2 Manos)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        # 🔹 Crear GIF resumen
        if frames_for_gif:
            gif_filename = f"{sign_name}_training.gif"
            self.gif_generator.create_gif(frames_for_gif, gif_filename)
            print(f"🎞️ GIF guardado: {gif_filename}")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
