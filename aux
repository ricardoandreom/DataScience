import numpy as np
from fastdtw import fastdtw

# Supondo que você tem um DataFrame chamado 'df' com os dados dos clientes
# e um DataFrame chamado 'grupo_df' com os dados dos grupos e seus descritivos

# 1. Calcular as séries temporais médias para cada grupo
series_temporais_grupos = {}
for grupo, clientes in grupo_df.groupby('grupo'):
    series_temporais = []
    for _, cliente in clientes.iterrows():
        # Suponha que 'cliente' é uma linha do DataFrame com as features do cliente
        # Aqui você pode pegar as features relevantes ou os valores SHAP e formar uma série temporal
        serie_temporal_cliente = np.array([cliente[feature] for feature in features_relevantes])
        series_temporais.append(serie_temporal_cliente)
    media_grupo = np.mean(series_temporais, axis=0)
    series_temporais_grupos[grupo] = media_grupo

# 2. Calcular a distância de DTW entre cada cliente e os grupos
for _, cliente in df.iterrows():
    serie_temporal_cliente = np.array([cliente[feature] for feature in features_relevantes])
    distancias_dtw = {}
    for grupo, serie_temporal_grupo in series_temporais_grupos.items():
        distancia, _ = fastdtw(serie_temporal_cliente, serie_temporal_grupo)
        distancias_dtw[grupo] = distancia
    grupo_mais_proximo = min(distancias_dtw, key=distancias_dtw.get)
    # Agora 'grupo_mais_proximo' contém o grupo mais próximo para o cliente atual
