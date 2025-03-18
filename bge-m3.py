import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import pandas as pd

# 모델과 토크나이저 로드
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3")
    return tokenizer, model

# 텍스트에 대한 임베딩 벡터 가져오기
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # [CLS] 토큰 임베딩을 문장 임베딩으로 사용
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

# 딕셔너리들의 임베딩 시각화 (Plotly 사용)
def visualize_embeddings_plotly(dict_list, method='pca', save_path=None):
    tokenizer, model = load_model()
    
    # 임베딩과 메타데이터를 저장할 리스트
    all_embeddings = []
    dict_indices = []
    keys = []
    texts = []
    
    # 딕셔너리의 각 항목에 대한 임베딩 구하기
    for dict_idx, data_dict in enumerate(dict_list):
        for key, text in data_dict.items():
            embedding = get_embedding(text, tokenizer, model)
            all_embeddings.append(embedding)
            dict_indices.append(dict_idx)
            keys.append(key)
            texts.append(text)
    
    # 배열로 변환
    embedding_array = np.array(all_embeddings)
    
    # 차원 축소 (PCA)
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("현재는 'pca' 방법만 지원합니다")
    
    reduced_embeddings = reducer.fit_transform(embedding_array)
    
    # DataFrame 생성 (Plotly용)
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Dictionary': [f"Dictionary {i}" for i in dict_indices],
        'Type': keys,
        'Text': texts,
        'Label': [f"Dict {i} - {k}" for i, k in zip(dict_indices, keys)]
    })
    
    # Plotly로 시각화
    fig = px.scatter(
        df, x='x', y='y', 
        color='Dictionary', 
        symbol='Type',
        text='Label',
        hover_data=['Text'],  # 호버 시 텍스트 표시
        title=f"BGE-M3 embedding ({method.upper()})",
        labels={'x': 'dim 1', 'y': 'dim 2'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # 추가 스타일링
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        width=1000,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # 그리드 추가
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # 저장 경로가 주어진 경우 그림 저장
    if save_path:
        fig.write_html(f"{save_path}.html")  # 인터랙티브 HTML로 저장
        fig.write_image(f"{save_path}.png")  # 정적 이미지로도 저장
    
    return fig

# 예시 사용
dict_list = []

with open("C:/Users/wjdrb/multilingual_embeddings/code-switch.json", "r", encoding="UTF-8") as f:
    dict_list = json.load(f)

# Plotly를 사용한 인터랙티브 임베딩 시각화
fig = visualize_embeddings_plotly(dict_list, method='pca', save_path='embeddings_interactive')
fig.show()