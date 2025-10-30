import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
    def detect_hands(self, image):
        """Detecta manos y devuelve landmarks"""
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_landmarks_list = []
        handedness_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_landmarks_list.append(hand_landmarks)
                handedness_list.append(handedness)
                
        return hand_landmarks_list, handedness_list, results
    
    def draw_landmarks(self, image, hand_landmarks):
        """Dibuja landmarks en la imagen"""
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
    
    def get_hand_roi(self, image, hand_landmarks, padding=20):
        """Obtiene ROI basado en landmarks de la mano"""
        h, w = image.shape[:2]
        landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
        
        x_min = int(np.min(landmarks_array[:, 0]) - padding)
        y_min = int(np.min(landmarks_array[:, 1]) - padding)
        x_max = int(np.max(landmarks_array[:, 0]) + padding)
        y_max = int(np.max(landmarks_array[:, 1]) + padding)
        
        # Asegurar que los valores están dentro de la imagen
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        return x_min, y_min, x_max, y_max
    
    def extract_landmarks_features(self, hand_landmarks, image_shape):
        """Extrae características numéricas de los landmarks"""
        h, w = image_shape[:2]
        features = []
        
        for lm in hand_landmarks.landmark:
            # Coordenadas normalizadas a píxeles
            x = lm.x * w
            y = lm.y * h
            z = lm.z * w  # Profundidad relativa
            
            features.extend([x, y, z])
            
        return np.array(features)
    
    def get_hand_angle_features(self, hand_landmarks):
        """Calcula ángulos entre joints para mejores características"""
        landmarks = hand_landmarks.landmark
        angles = []
        
        # Definir conexiones importantes para ángulos
        connections = [
            (0, 1, 2),   # Pulgar base
            (1, 2, 3),
            (2, 3, 4),
            (0, 5, 6),   # Índice
            (5, 6, 7),
            (6, 7, 8),
            (0, 9, 10),  # Medio
            (9, 10, 11),
            (10, 11, 12),
            (0, 13, 14), # Anular
            (13, 14, 15),
            (14, 15, 16),
            (0, 17, 18), # Meñique
            (17, 18, 19),
            (18, 19, 20)
        ]
        
        for i, j, k in connections:
            v1 = np.array([landmarks[i].x - landmarks[j].x, 
                          landmarks[i].y - landmarks[j].y])
            v2 = np.array([landmarks[k].x - landmarks[j].x, 
                          landmarks[k].y - landmarks[j].y])
            
            # Calcular ángulo
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            angles.append(angle)
            
        return np.array(angles)