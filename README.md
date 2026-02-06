# Classificador de Racas de Cachorros

Este projeto consiste em um sistema de visão computacional capaz de identificar raças de cachorros a partir de imagens. O modelo utiliza a técnica de Transfer Learning com a arquitetura MobileNetV2 (pré-treinada no ImageNet) e foi ajustado utilizando o Stanford Dogs Dataset.

O sistema verifica automaticamente a existência de um modelo treinado (neste pacote já existe um modelo treinado, caso deseje treinar outro modelo, apague o arquivo .keras e rode a aplicação novamente). Caso não encontre, ele realiza o treinamento e salva o arquivo para usos futuros. Caso encontre, ele carrega os pesos e disponibiliza uma interface via terminal para realizar predicoes em novas imagens. 

## Requisitos

Para executar este projeto, é necessário ter o Python instalado (versão 3.8 ou superior recomendada) e as seguintes bibliotecas:

* tensorflow
* numpy
* matplotlib
* pillow

Você pode instalar todas as dependências com o comando:

pip install tensorflow numpy matplotlib pillow

## Configuração do Dataset

O projeto foi configurado para utilizar o **Stanford Dogs Dataset**. Devido ao tamanho dos arquivos, as imagens não estão incluídas neste repositório.

1.  Baixe o dataset oficial (Stanford Dogs).
2.  Extraia o conteúdo.
3.  Crie uma pasta chamada `dataset` na raiz deste projeto.
4.  Mova a pasta `Images` (que contém as subpastas das raças) para dentro de `dataset`.

A estrutura final de pastas deve ficar assim:

projeto/
├── main.py
├── dataset/
│   └── Images/
│       ├── n02085620-Chihuahua/
│       ├── n02099601-Golden_retriever/
│       └── ... (outras pastas de raças)
└── ...

## Como Usar

1.  Abra o terminal na pasta do projeto.
2.  Execute o script principal:

python main.py

3.  **Primeira Execução:** O sistema identificará que não há um modelo salvo (`modelo_cachorros.keras`) e iniciará o processo de treinamento automaticamente. Isso pode levar alguns minutos dependendo do seu hardware.
4.  **Uso Normal:** Após o treinamento (ou se o modelo já existir), o programa solicitará o caminho de uma imagem.
5.  Digite o caminho completo da imagem que deseja analisar (exemplo: `/home/usuario/imagens/meu_cachorro.jpg`) e pressione Enter.
6.  O sistema exibirá a raça predita, a porcentagem de confiança e abrirá uma janela com a imagem classificada.
7.  Para encerrar o programa, digite `sair` quando for solicitado o caminho da imagem.

## Estrutura do Projeto

* **main.py:** Arquivo principal contendo todo o código fonte para treinamento, carregamento do modelo e inferência (predição).
* **modelo_cachorros.keras:** Arquivo gerado após o primeiro treinamento, contendo a rede neural compilada.
* **dataset/:** Diretório destinado ao armazenamento das imagens para treino e validação.
