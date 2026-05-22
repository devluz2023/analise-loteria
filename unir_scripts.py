import os

def unir_arquivos_python(diretorio_origem='.', arquivo_saida='todos_os_scripts.txt'):
    # Lista para armazenar o caminho dos arquivos encontrados
    arquivos_encontrados = []
    
    # Nomes de pastas que devem ser estritamente ignoradas
    pastas_para_ignorar = {'.venv', 'venv', '__pycache__', '.git', '.idea', '.vscode'}
    
    print("A analisar os diretórios (filtragem estrita ativada)...")
    
    # Varre o diretório e subdiretórios em busca de arquivos .py
    for raiz, diretorios, arquivos in os.walk(diretorio_origem):
        
        # Divide o caminho da pasta atual em partes para verificar se ela pertence ao venv ou pycache
        partes_do_caminho = set(raiz.replace('\\', '/').split('/'))
        
        # Se o caminho atual contiver qualquer uma das pastas proibidas, ignora completamente
        if partes_do_caminho.intersection(pastas_para_ignorar):
            continue
            
        for arquivo in arquivos:
            # Garante que vai ler apenas arquivos .py legítimos
            if arquivo.endswith('.py') and arquivo != os.path.basename(__file__):
                caminho_completo = os.path.join(raiz, arquivo)
                arquivos_encontrados.append(caminho_completo)
    
    if not arquivos_encontrados:
        print("Nenhum arquivo Python (.py) seu foi encontrado.")
        return
        
    print(f"Encontrados {len(arquivos_encontrados)} arquivos .py legítimos. A iniciar a união...")
    
    # Abre o arquivo TXT de saída com codificação UTF-8
    with open(arquivo_saida, 'w', encoding='utf-8') as f_saida:
        for caminho in sorted(arquivos_encontrados):
            nome_arquivo = os.path.basename(caminho)
            
            # Escreve um cabeçalho bem visível separando cada script
            f_saida.write("=" * 80 + "\n")
            f_saida.write(f" ARQUIVO: {nome_arquivo}\n")
            f_saida.write(f" CAMINHO: {caminho}\n")
            f_saida.write("=" * 80 + "\n\n")
            
            try:
                # Lê o conteúdo do script e escreve no arquivo unificado
                with open(caminho, 'r', encoding='utf-8') as f_entrada:
                    conteudo = f_entrada.read()
                    f_saida.write(conteudo)
            except Exception as e:
                f_saida.write(f"[ERRO AO LER ESTE ARQUIVO]: {str(e)}\n")
                print(f"Aviso: Não foi possível ler o arquivo {nome_arquivo}.")
                
            f_saida.write("\n\n") # Espaçamento entre scripts
            
    print(f"Sucesso! Todos os seus scripts foram unidos em: '{arquivo_saida}'")
    print("Apenas o seu código real foi guardado. .venv e __pycache__ foram 100% bloqueados.")

if __name__ == '__main__':
    unir_arquivos_python()