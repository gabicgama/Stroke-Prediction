# stroke-prediction

## Trabalho realizado como conclusão de curso de Engenharia de Computação :mortar_board:

O projeto se resume ao desenvolvimento de um modelo de machine learning aplicando a técnica de Stacked Generalization e implantação em uma aplicação web para uso com novos dados.

- Modelo desenvolvido por meio do Jupyter Notebook
    - Bibliotecas utilizadas: Scikit-learn, Pandas, Numpy, Seaborn, Matplotlib, GXBoost
- Exportação dos modelos utilizando a biblioteca pickle
- Disponibilização do modelo desenvolvendo uma aplicação web 
    - API  desenvolvida com Flask e Flasgger
    - Deploy na web pode ser feito com Google Cloud Platform (free tier)

Fixes para GCP:

- Executar ```export LD_LIBRARY_PATH=/usr/local/lib ```
