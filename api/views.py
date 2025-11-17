# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json

# @csrf_exempt
# def predict(request):
    
#     if request.method == 'POST':
        
#         try:
#             data = json.loads(request.body) 
#             print("Dados recebidos do frontend:", data)

#         except json.JSONDecodeError:
#             return JsonResponse({"error": "JSON inválido"}, status=400)

#         mock_response = {
#             "default": 0,
#             "probability": 0.85
#         }
        
#         return JsonResponse(mock_response, status=200)
    
#     else:
#         return JsonResponse({"error": "Método não permitido. Use POST."}, status=405)

# api/views.py

import xgboost as xgb
import numpy as np
import pandas as pd
import joblib  # Para carregar o scaler.joblib
import json    # Para carregar o feature_columns.json
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

# --- 1. Definir Caminhos dos Artefatos ---
# (Assumindo que os 3 arquivos estão na pasta 'api')
MODEL_PATH = os.path.join(settings.BASE_DIR, 'api', 'modelo_inadimplencia.json')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'api', 'scaler.joblib')
COLUMNS_PATH = os.path.join(settings.BASE_DIR, 'api', 'feature_columns.json')

# --- 2. Carregar os Artefatos (Quando o servidor inicia) ---
try:
    # Carregar o Modelo XGBoost
    bst = xgb.XGBClassifier()
    bst.load_model(MODEL_PATH)
    
    # Carregar o Scaler (StandardScaler)
    scaler = joblib.load(SCALER_PATH)
    
    # Carregar a Lista de Colunas de Features
    with open(COLUMNS_PATH, 'r') as f:
        feature_columns = json.load(f) # Lista de 58 colunas
        
    print(f"Sucesso: Modelo, Scaler e Lista de Colunas ({len(feature_columns)} colunas) carregados.")
    
except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR ARTEFATOS: {e}")
    bst = None
    scaler = None
    feature_columns = None

# --- 3. Função de Pré-processamento ---
def preprocess_input(data):
    """
    Aplica as *mesmas* transformações do notebook 'modelo_smote.py'
    em um novo dado de entrada (JSON).
    """
    # Converter o JSON de entrada em um DataFrame de 1 linha
    df = pd.DataFrame([data])

    # --- Aplicar transformações do notebook ---
    
    # 1. 'Verification Status': Substituir 'Source Verified' por 'Verified'
    if df.get('Verification Status', '').iloc[0] == 'Source Verified':
        df.loc[0, 'Verification Status'] = 'Verified'
            
    # 2. 'Interest Rate': Arredondar para cima
    if 'Interest Rate' in df.columns:
        df['Interest Rate'] = np.ceil(df['Interest Rate'])
        
    # 3. 'Balance': Aplicar logaritmo
    if 'Balance' in df.columns:
        df['Balance'] = np.log(df['Balance'] + 1e-6) # +1e-6 para evitar log(0)

    # 4. Aplicar Dummies (One-Hot Encoding)
    vc = ['Grade', 'Sub Grade', 'HomeOwnership', 'Initial List Status', 'Verification Status']
    vc_present = [col for col in vc if col in df.columns]
    
    if vc_present:
        df = pd.get_dummies(data=df, columns=vc_present, prefix='Col', drop_first=True, prefix_sep="_", dtype='int8')

    # 5. Alinhar Colunas (O passo mais importante)
    # Garante que o DataFrame tenha as *exatas* 58 colunas que o modelo espera,
    # na ordem correta, preenchendo com 0 as que faltarem.
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df

# --- 4. Endpoint da API ---
@api_view(['POST'])
def predict_inadimplencia(request):
    """
    Endpoint para prever a inadimplência.
    Recebe um JSON com as features BRUTAS (antes do processamento).
    """
    # Verifica se os artefatos foram carregados
    if not all([bst, scaler, feature_columns]):
        return Response(
            {"erro": "Serviço indisponível: Falha ao carregar artefatos do modelo."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    try:
        # 1. Pegar os dados brutos do POST
        raw_data = request.data

        # 2. Pré-processar os dados brutos
        df_processed = preprocess_input(raw_data)
        
        # 3. Escalar os dados com o scaler
        features_scaled = scaler.transform(df_processed)

        # 4. Fazer a predição
        prediction = bst.predict(features_scaled)
        probability = bst.predict_proba(features_scaled)

        # 5. Formatar e retornar a resposta
        response_data = {
            'predicao_classe': int(prediction[0]), # 0 ou 1
            'probabilidade_inadimplencia': float(probability[0][1]) # Prob de ser classe 1
        }
        
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"erro": f"Ocorreu um erro durante a predição: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )