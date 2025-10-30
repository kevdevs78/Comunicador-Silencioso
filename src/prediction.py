import cv2
import numpy as np
import joblib
import os
from src.hand_detector import HandDetector
from src.text_to_speech import TextToSpeech
from src.gif_generator import GIFGenerator

class Predictor:
    def __init__(self):
        self.model_path = "data/trained_models/sign_language_model.pkl"
        self.class_names_path = "data/trained_models/class_names.pkl"
        self.scaler_path = "data/trained_models/scaler.pkl"
        
        self.cap = cv2.VideoCapture(0)
        self.hand_detector = HandDetector(max_num_hands=2)  # ✅ Detectar hasta 2 manos
        self.tts = TextToSpeech()
        self.gif_generator = GIFGenerator()
        
        # Cargar modelo entrenado
        try:
            self.model = joblib.load(self.model_path)
            self.class_names = joblib.load(self.class_names_path)
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ Modelo cargado. Clases: {self.class_names}")
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            print("⚠️ Entrena el modelo primero.")
            self.model = None
    
    def predict_real_time(self):
        if self.model is None:
            return
        
        frames_for_gif = []
        current_prediction = ""
        confidence_threshold = 0.6
        prediction_stability = []
        stability_threshold = 5
        
        print("\n=== MODO PREDICCIÓN (2 MANOS) ===")
        print("Presiona 's' para reproducir voz, 'g' para guardar GIF, 'q' para salir")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # 🔹 Detectar manos (una o dos)
            hand_landmarks_list, handedness_list, _ = self.hand_detector.detect_hands(frame)
            all_features = []

            if hand_landmarks_list:
                # Procesar cada mano detectada
                for i, hand_landmarks in enumerate(hand_landmarks_list):
                    handedness = handedness_list[i]
                    self.hand_detector.draw_landmarks(display_frame, hand_landmarks)

                    # Extraer características
                    landmarks_features = self.hand_detector.extract_landmarks_features(hand_landmarks, frame.shape)
                    angle_features = self.hand_detector.get_hand_angle_features(hand_landmarks)
                    hand_features = np.concatenate([landmarks_features, angle_features])
                    all_features.extend(hand_features)

                    # Dibujar rectángulo ROI
                    x_min, y_min, x_max, y_max = self.hand_detector.get_hand_roi(frame, hand_landmarks, padding=30)
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Mano {i+1}: {handedness.classification[0].label}",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 🔹 Ajustar tamaño de features según el modelo entrenado
                expected_features = self.scaler.mean_.shape[0]
                current_features = len(all_features)

                if current_features < expected_features:
                    # Rellenar con ceros si solo hay una mano
                    padding = expected_features - current_features
                    all_features = np.concatenate([all_features, np.zeros(padding)])
                elif current_features > expected_features:
                    # Recortar si hay más features
                    all_features = all_features[:expected_features]

                # Escalar y predecir
                features_scaled = self.scaler.transform([all_features])
                probabilities = self.model.predict_proba(features_scaled)[0]
                max_prob = np.max(probabilities)
                predicted_class = np.argmax(probabilities)

                if max_prob > confidence_threshold:
                    new_prediction = self.class_names[predicted_class]
                    prediction_stability.append(new_prediction)
                    if len(prediction_stability) > stability_threshold:
                        prediction_stability.pop(0)

                    # Verificar estabilidad
                    if len(prediction_stability) == stability_threshold and all(p == new_prediction for p in prediction_stability):
                        current_prediction = new_prediction

                    # Mostrar predicción
                    color = (0, 255, 0) if current_prediction == new_prediction else (0, 255, 255)
                    cv2.putText(display_frame, f"Seña: {current_prediction}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(display_frame, f"Confianza: {max_prob:.3f}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display_frame, "Seña no reconocida",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                cv2.putText(display_frame, "Mostrar manos a la cámara",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                current_prediction = ""
                prediction_stability = []

            # Guardar frames para el GIF
            if len(frames_for_gif) % 3 == 0:
                gif_frame = cv2.resize(display_frame, (320, 240))
                frames_for_gif.append(gif_frame)

            cv2.imshow("Traductor de Señas - Predicción", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and current_prediction:
                self.tts.speak(current_prediction)
            elif key == ord('g') and current_prediction and frames_for_gif:
                self.gif_generator.create_prediction_gif(frames_for_gif[-15:], current_prediction)
                print(f"🎞️ GIF guardado para: {current_prediction}")

        # Guardar GIF final al salir
        if frames_for_gif:
            self.gif_generator.create_gif(frames_for_gif[-30:], "session_final.gif")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
