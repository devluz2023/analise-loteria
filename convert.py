import os
from moviepy.editor import VideoFileClip

def converter_mov_para_mp4(caminho_pasta):
    # Verifica se a pasta existe
    if not os.path.exists(caminho_pasta):
        print(f"Erro: A pasta '{caminho_pasta}' não foi encontrada.")
        return

    # Lista todos os arquivos na pasta
    arquivos = [f for f in os.listdir(caminho_pasta) if f.lower().endswith(".mov")]
    
    if not arquivos:
        print("Nenhum arquivo .mov encontrado na pasta.")
        return

    print(f"Encontrados {len(arquivos)} arquivos para converter...")

    for arquivo in arquivos:
        caminho_entrada = os.path.join(caminho_pasta, arquivo)
        nome_saida = os.path.splitext(arquivo)[0] + ".mp4"
        caminho_saida = os.path.join(caminho_pasta, nome_saida)
        
        print(f"Convertendo: {arquivo} -> {nome_saida}")
        
        try:
            # Carrega o vídeo e exporta como mp4
            clip = VideoFileClip(caminho_entrada)
            clip.write_videofile(caminho_saida, codec="libx264", audio_codec="aac")
            clip.close()
        except Exception as e:
            print(f"Erro ao converter {arquivo}: {e}")

    print("\nProcesso concluído!")

# Caminho da sua pasta
pasta_alvo = '/Users/fabiojuliodaluz/Documents/video'
converter_mov_para_mp4(pasta_alvo)