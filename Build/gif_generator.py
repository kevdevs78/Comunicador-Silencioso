import cv2
import os
from PIL import Image
import numpy as np

class GIFGenerator:
    def __init__(self):
        self.output_path = "data/gifs"
        os.makedirs(self.output_path, exist_ok=True)
    
    def create_gif(self, frames, filename, duration=100):
        """
        Crea un GIF a partir de una lista de frames
        frames: lista de frames en formato BGR (OpenCV)
        filename: nombre del archivo GIF
        duration: duración de cada frame en milisegundos
        """
        try:
            # Convertir frames BGR a RGB
            rgb_frames = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                rgb_frames.append(pil_image)
            
            # Guardar como GIF
            gif_path = os.path.join(self.output_path, filename)
            
            # Guardar el primer frame y luego append los demás
            if rgb_frames:
                rgb_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=rgb_frames[1:],
                    duration=duration,
                    loop=0  # loop infinito
                )
            
            print(f"GIF guardado exitosamente: {gif_path}")
            return gif_path
            
        except Exception as e:
            print(f"Error al crear GIF: {e}")
            return None
    
    def create_prediction_gif(self, frames, predicted_text):
        """
        Crea un GIF especial para predicciones con texto overlay
        """
        try:
            # Agregar texto a cada frame
            frames_with_text = []
            for frame in frames:
                # Agregar texto de predicción
                text_frame = frame.copy()
                cv2.putText(text_frame, f"Seña: {predicted_text}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames_with_text.append(text_frame)
            
            filename = f"prediction_{predicted_text}.gif"
            return self.create_gif(frames_with_text, filename)
            
        except Exception as e:
            print(f"Error al crear GIF de predicción: {e}")
            return None