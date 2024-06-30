import cv2
import mediapipe as mp
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import pairwise

class DetectorDeRostos:
    def __init__(self, pasta_base_dados="rostos_detectados", pasta_imagens="imagens"):
        self.reconhecimento_rosto = mp.solutions.face_detection
        self.desenho = mp.solutions.drawing_utils
        self.reconhecedor_rosto = self.reconhecimento_rosto.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.pasta_base_dados = pasta_base_dados
        self.pasta_imagens = pasta_imagens

        # Criar pasta base de dados se não existir
        if not os.path.exists(self.pasta_base_dados):
            os.makedirs(self.pasta_base_dados)

        # Criar pasta de imagens se não existir
        if not os.path.exists(self.pasta_imagens):
            os.makedirs(self.pasta_imagens)

        # Carregar rostos conhecidos
        self.rostos_conhecidos, self.nomes_rostos_conhecidos = self.carregar_rostos_conhecidos()

    def carregar_rostos_conhecidos(self):
        rostos_conhecidos = []
        nomes_rostos_conhecidos = []
        for nome_arquivo in os.listdir(self.pasta_base_dados):
            caminho_arquivo = os.path.join(self.pasta_base_dados, nome_arquivo)
            imagem = cv2.imread(caminho_arquivo)
            if imagem is not None:
                rostos_conhecidos.append(self.ajustar_tamanho_rosto(imagem))
                nomes_rostos_conhecidos.append(nome_arquivo)
        return rostos_conhecidos, nomes_rostos_conhecidos

    def ajustar_tamanho_rosto(self, rosto):
        rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
        rosto = cv2.resize(rosto, (150, 150))
        return rosto

    def salvar_rosto(self, imagem, bounding_box):
        altura, largura, _ = imagem.shape
        x_min = int(bounding_box.xmin * largura)
        y_min = int(bounding_box.ymin * altura)
        x_max = x_min + int(bounding_box.width * largura)
        y_max = y_min + int(bounding_box.height * altura)
        rosto = imagem[y_min:y_max, x_min:x_max]
        rosto_cinza = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
        rosto_redimensionado = cv2.resize(rosto_cinza, (150, 150))

        # Solicitar nome do usuário
        nome_pessoa = input("Digite o nome da pessoa: ")

        caminho_arquivo = os.path.join(self.pasta_base_dados, f"{nome_pessoa}.png")
        cv2.imwrite(caminho_arquivo, rosto_redimensionado)
        print(f"Rosto salvo em: {caminho_arquivo}")

        self.rostos_conhecidos.append(rosto_redimensionado)
        self.nomes_rostos_conhecidos.append(nome_pessoa)

    def reconhecer_rosto(self, rosto_desconhecido):
        if not self.rostos_conhecidos:
            return False, None

        rosto_desconhecido = self.ajustar_tamanho_rosto(rosto_desconhecido)
        rostos_conhecidos_array = np.array([rosto for rosto in self.rostos_conhecidos])

        distancias = pairwise.euclidean_distances(rosto_desconhecido.flatten().reshape(1, -1), rostos_conhecidos_array.reshape(len(self.rostos_conhecidos), -1))

        limiar = 5000
        indice_menor_distancia = np.argmin(distancias)
        menor_distancia = distancias[0][indice_menor_distancia]

        if menor_distancia < limiar:
            return True, self.nomes_rostos_conhecidos[indice_menor_distancia]
        else:
            return False, None

    def executar(self):
        for nome_arquivo in os.listdir(self.pasta_imagens):
            caminho_arquivo = os.path.join(self.pasta_imagens, nome_arquivo)
            frame = cv2.imread(caminho_arquivo)
            if frame is None:
                continue

            imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lista_rostos = self.reconhecedor_rosto.process(imagem)
            imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)

            if lista_rostos.detections:
                for rosto in lista_rostos.detections:
                    bounding_box = rosto.location_data.relative_bounding_box
                    x_min = int(bounding_box.xmin * frame.shape[1])
                    y_min = int(bounding_box.ymin * frame.shape[0])
                    x_max = x_min + int(bounding_box.width * frame.shape[1])
                    y_max = y_min + int(bounding_box.height * frame.shape[0])

                    if x_min >= 0 and y_min >= 0 and x_max <= frame.shape[1] and y_max <= frame.shape[0]:
                        rosto_desconhecido = frame[y_min:y_max, x_min:x_max]

                        reconhecido, nome_pessoa = self.reconhecer_rosto(rosto_desconhecido)
                        if reconhecido:
                            cv2.putText(imagem, f"Reconhecido: {nome_pessoa}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            print(f"Rosto reconhecido: {nome_pessoa}")
                        else:
                            self.salvar_rosto(frame, bounding_box)
                            cv2.putText(imagem, "Novo rosto salvo", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            print("Rosto novo salvo na base de dados.")

                        self.desenho.draw_detection(imagem, rosto)

            _, im_buf_arr = cv2.imencode(".jpg", imagem)
            from IPython.display import display, Image
            display(Image(data=im_buf_arr.tobytes()))
            cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DetectorDeRostos()
    detector.executar().