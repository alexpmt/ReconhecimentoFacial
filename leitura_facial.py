import cv2
import mediapipe as mp
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

class DetectorDeRostos:
    def __init__(self, pasta_base_dados="Rostos_Salvos", pasta_presenca="presença", modelo_url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"):
        self.webcam = cv2.VideoCapture(0)  
        self.reconhecimento_rosto = mp.solutions.face_detection  
        self.desenho = mp.solutions.drawing_utils  
        self.reconhecedor_rosto = self.reconhecimento_rosto.FaceDetection()  
        self.pasta_base_dados = pasta_base_dados
        self.pasta_presenca = pasta_presenca

        if not os.path.exists(self.pasta_base_dados):
            os.makedirs(self.pasta_base_dados)

        if not os.path.exists(self.pasta_presenca):
            os.makedirs(self.pasta_presenca)

        self.modelo = hub.load(modelo_url)

        self.embeddings_conhecidos = []
        self.nomes_conhecidos = []
        self.carregar_embeddings()

        self.nome_pessoa_presenca = None

        self.presencas_registradas = {}

    def preprocessar_rosto(self, rosto):
        rosto = cv2.resize(rosto, (224, 224))  
        rosto = rosto.astype('float32') / 255.0  
        rosto = np.expand_dims(rosto, axis=0)  
        return rosto

    def extrair_embedding(self, rosto):
        rosto_preprocessado = self.preprocessar_rosto(rosto)
        embedding = self.modelo(rosto_preprocessado)
        return embedding.numpy()

    def salvar_rosto(self, imagem, bounding_box):
        altura, largura, _ = imagem.shape
        x_min = int(bounding_box.xmin * largura)
        y_min = int(bounding_box.ymin * altura)
        x_max = x_min + int(bounding_box.width * largura)
        y_max = y_min + int(bounding_box.height * altura)
        rosto = imagem[y_min:y_max, x_min:x_max]

        embedding = self.extrair_embedding(rosto)

        reconhecido, nome = self.verificar_reconhecimento(embedding)
        if reconhecido:
            print(f"Rosto reconhecido como: {nome}")
            if nome not in self.presencas_registradas:
                self.registrar_presenca(nome)
                self.presencas_registradas[nome] = True
            return False  
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            caminho_arquivo = os.path.join(self.pasta_base_dados, f"rosto_{timestamp}.png")

            cv2.imwrite(caminho_arquivo, rosto)
            print(f"Rosto salvo em: {caminho_arquivo}")

            np.save(caminho_arquivo.replace(".png", ".npy"), embedding)
            print(f"Embedding salvo em: {caminho_arquivo.replace('.png', '.npy')}")

            self.embeddings_conhecidos.append(embedding)
            self.nomes_conhecidos.append(self.nome_pessoa_presenca)

            self.registrar_presenca(self.nome_pessoa_presenca)
            self.presencas_registradas[self.nome_pessoa_presenca] = True
            return True

    def carregar_embeddings(self):
        for arquivo in os.listdir(self.pasta_base_dados):
            if arquivo.endswith(".npy"):
                caminho_arquivo = os.path.join(self.pasta_base_dados, arquivo)
                embedding = np.load(caminho_arquivo)
                self.embeddings_conhecidos.append(embedding)
                nome = arquivo.split("_")[-1].replace(".npy", "")
                self.nomes_conhecidos.append(nome)

    def verificar_reconhecimento(self, embedding):
        for i, emb_conhecido in enumerate(self.embeddings_conhecidos):
            similaridade = cosine_similarity(embedding.reshape(1, -1), emb_conhecido.reshape(1, -1))
            if similaridade > 0.7:  
                return True, self.nomes_conhecidos[i]
        return False, None

    def registrar_presenca(self, nome):
        timestamp = datetime.now().strftime("%H:%M")
        caminho_arquivo = os.path.join(self.pasta_presenca, "presenca.txt")

        with open(caminho_arquivo, "a") as file:
            file.write(f"{nome} está presente. Registro feito em: {timestamp}\n")
        print(f"Presença de {nome} registrada em: {caminho_arquivo}")

    def executar(self, nome_pessoa_presenca):
        self.nome_pessoa_presenca = nome_pessoa_presenca
        while self.webcam.isOpened():
            validacao, frame = self.webcam.read()  
            if not validacao:
                break
            imagem = frame
            lista_rostos = self.reconhecedor_rosto.process(imagem)  
            
            if lista_rostos.detections:  
                for rosto in lista_rostos.detections:  
                    self.desenho.draw_detection(imagem, rosto)  
                    if self.nome_pessoa_presenca is not None:
                        self.salvar_rosto(imagem, rosto.location_data.relative_bounding_box)

            cv2.imshow("Rostos na sua webcam", imagem) 
            if cv2.waitKey(5) == 27:  
                break

        self.webcam.release()
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    nome_pessoa_presenca = input("Digite seu nome para registro de presença: ")
    detector = DetectorDeRostos(pasta_presenca="presença")
    detector.executar(nome_pessoa_presenca)
