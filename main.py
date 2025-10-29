import cv2
import numpy as np
import os
from src.data_collection import DataCollector
from src.model_training import ModelTrainer
from src.prediction import Predictor
from src.text_to_speech import TextToSpeech
from src.gif_generator import GIFGenerator

class SignLanguageTranslator:
    def __init__(self):
        self.data_collector = DataCollector()
        self.model_trainer = ModelTrainer()
        self.predictor = Predictor()
        self.tts = TextToSpeech()
        self.gif_generator = GIFGenerator()
        
    def show_menu(self):
        print("\n=== TRADUCTOR DE LENGUAJE DE SEÑAS ===")
        print("1. Entrenar modelo (capturar señas y guardar como GIF)")
        print("2. Ejecutar predicción en tiempo real")
        print("3. Salir")
        
    def run(self):
        while True:
            self.show_menu()
            choice = input("Selecciona una opción: ")
            
            if choice == "1":
                self.train_model()
            elif choice == "2":
                self.predict_real_time()
            elif choice == "3":
                print("¡Hasta luego!")
                break
            else:
                print("Opción inválida")
    
    def train_model(self):
        print("\n=== MODO ENTRENAMIENTO ===")
        sign_name = input("Nombre de la seña (ej: 'hola', 'gracias'): ")
        num_samples = int(input("Número de muestras a capturar: "))
        
        self.data_collector.collect_data(sign_name, num_samples)
        self.model_trainer.train_model()
        print("✅ ¡Entrenamiento completado!")
    
    def predict_real_time(self):
        print("\n=== MODO PREDICCIÓN ===")
        print("Presiona 'q' para salir, 's' para reproducir voz")
        self.predictor.predict_real_time()

if __name__ == "__main__":
    translator = SignLanguageTranslator()
    translator.run()
