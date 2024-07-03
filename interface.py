import tkinter as tk
import tkinter.messagebox as messagebox
from leitura_facial import DetectorDeRostos
from converter_audio_para_texto import reconhecer_nome_por_voz


class AplicacaoTkinter:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Rostos")

        self.label_instrucao = tk.Label(self.root, text="Clique no botão para iniciar o reconhecimento por voz:")
        self.label_instrucao.pack(pady=10)

        self.botao_iniciar = tk.Button(self.root, text="Iniciar Reconhecimento por Voz",
                                       command=self.iniciar_reconhecimento_voz)
        self.botao_iniciar.pack(pady=20)

    def iniciar_reconhecimento_voz(self):
        nome_pessoa_presenca = reconhecer_nome_por_voz()

        if nome_pessoa_presenca:
            messagebox.showinfo("Detecção de Rostos", f"Detecção de rostos iniciada para {nome_pessoa_presenca}.")
            detector = DetectorDeRostos(pasta_presenca="presença")
            detector.executar(nome_pessoa_presenca)
        else:
            messagebox.showerror("Erro", "Não foi possível reconhecer seu nome. Tente novamente.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacaoTkinter(root)
    root.mainloop()
