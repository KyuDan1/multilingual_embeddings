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

# 딕셔너리들의 임베딩 시각화 (Plotly 사용) - 선택한 딕셔너리만 하이라이트
def visualize_embeddings_plotly(dict_list, method='pca', save_path=None, selected_dict_idx=None):
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
    
    # 선택된 딕셔너리에 따라 마커 크기와 투명도 설정
    marker_size = []
    marker_opacity = []
    for idx in dict_indices:
        if selected_dict_idx is None or idx == selected_dict_idx:
            marker_size.append(15)  # 선택된 딕셔너리 또는 전체 보기 모드일 때 큰 마커
            marker_opacity.append(1.0)  # 선택된 딕셔너리 또는 전체 보기 모드일 때 불투명
        else:
            marker_size.append(8)  # 선택되지 않은 딕셔너리는 작은 마커
            marker_opacity.append(0.3)  # 선택되지 않은 딕셔너리는 반투명
    
    df['marker_size'] = marker_size
    df['marker_opacity'] = marker_opacity
    
    # 타이틀 설정
    if selected_dict_idx is not None:
        title = f"BGE-M3 embedding ({method.upper()}) - Dictionary {selected_dict_idx} Highlighted"
    else:
        title = f"BGE-M3 embedding ({method.upper()}) with convex polygon connections"
    
    # Plotly로 시각화 - 기본 산점도
    fig = px.scatter(
        df, x='x', y='y', 
        color='Dictionary', 
        symbol='Type',
        text='Label',
        hover_data=['Text'],
        title=title,
        labels={'x': 'dim 1', 'y': 'dim 2'},
        color_discrete_sequence=px.colors.qualitative.Set1,
        size_max=15  # 최대 마커 크기
    )
    
    # 마커 크기와 투명도 설정
    for i, trace in enumerate(fig.data):
        trace.marker.size = df[df['Dictionary'] == trace.name]['marker_size']
        trace.marker.opacity = df[df['Dictionary'] == trace.name]['marker_opacity']
        trace.marker.line = dict(width=1, color='DarkSlateGrey')
    
    # 각 딕셔너리 내의 점들을 선으로 연결
    for dict_idx in range(len(dict_list)):
        # 현재 딕셔너리의 선 색상과 두께 설정
        line_color = 'red' if (selected_dict_idx is None or dict_idx == selected_dict_idx) else 'lightgrey'
        line_width = 2.0 if (selected_dict_idx is None or dict_idx == selected_dict_idx) else 0.8
        line_opacity = 1.0 if (selected_dict_idx is None or dict_idx == selected_dict_idx) else 0.3
        
        # 현재 딕셔너리에 속하는 점들 필터링
        dict_points = df[df['Dictionary'] == f"Dictionary {dict_idx}"]
        
        # 딕셔너리 내의 모든 타입 (EtoK, KtoE, English, Korean)
        types = ['EtoK', 'KtoE', 'English', 'Korean']
        
        # 각 타입별로 좌표 찾기
        points = {}
        for type_name in types:
            points_of_type = dict_points[dict_points['Type'] == type_name]
            if not points_of_type.empty:
                points[type_name] = {
                    'x': points_of_type['x'].values[0],
                    'y': points_of_type['y'].values[0],
                    'idx': points_of_type.index[0]
                }
        
        # 점이 2개 이상 있을 때만 처리
        if len(points) >= 3:
            # 볼록 사각형(convex hull)을 만들기 위해 점들을 정렬
            # 중심점 계산
            center_x = sum(p['x'] for p in points.values()) / len(points)
            center_y = sum(p['y'] for p in points.values()) / len(points)
            
            # 중심점에서의 각도를 기준으로 정렬
            sorted_types = sorted(points.keys(), 
                                 key=lambda t: np.arctan2(points[t]['y'] - center_y, 
                                                         points[t]['x'] - center_x))
            
            # 정렬된 순서로 점들을 연결하여 볼록 다각형 생성
            for i in range(len(sorted_types)):
                current_type = sorted_types[i]
                next_type = sorted_types[(i + 1) % len(sorted_types)]
                
                # 연결선 추가
                fig.add_trace(go.Scatter(
                    x=[points[current_type]['x'], points[next_type]['x']],
                    y=[points[current_type]['y'], points[next_type]['y']],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash=None),
                    opacity=line_opacity,
                    showlegend=False,
                    hoverinfo='none'  # 호버 정보 비활성화
                ))
            
            # 마지막 점과 첫 번째 점 연결 (다각형 완성)
            if len(sorted_types) >= 2 and len(sorted_types) < 4:
                fig.add_trace(go.Scatter(
                    x=[points[sorted_types[-1]]['x'], points[sorted_types[0]]['x']],
                    y=[points[sorted_types[-1]]['y'], points[sorted_types[0]]['y']],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash=None),
                    opacity=line_opacity,
                    showlegend=False,
                    hoverinfo='none'
                ))
    
    # 레이아웃 설정
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=1500,
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
        if selected_dict_idx is not None:
            save_path = f"{save_path}_dict{selected_dict_idx}"
        fig.write_html(f"{save_path}.html")  # 인터랙티브 HTML로 저장
        fig.write_image(f"{save_path}.png")  # 정적 이미지로도 저장
    
    return fig

# 딕셔너리 선택 UI를 포함한 시각화 함수
def visualize_with_dictionary_selection(dict_list, method='pca', save_path=None):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    # 딕셔너리 수 계산
    num_dicts = len(dict_list)
    
    # 드롭다운 옵션 생성
    options = [('모든 딕셔너리 보기', None)] + [(f'Dictionary {i}', i) for i in range(num_dicts)]
    
    # 드롭다운 위젯 생성
    dropdown = widgets.Dropdown(
        options=options,
        value=None,
        description='선택:',
        style={'description_width': 'initial'}
    )
    
    # 출력 영역
    output = widgets.Output()
    
    # 드롭다운 변경 시 실행할 함수
    def on_change(change):
        with output:
            clear_output(wait=True)
            selected_dict = change['new']
            fig = visualize_embeddings_plotly(dict_list, method=method, save_path=save_path, selected_dict_idx=selected_dict)
            fig.show()
    
    # 변경 이벤트 연결
    dropdown.observe(on_change, names='value')
    
    # 초기 시각화
    with output:
        fig = visualize_embeddings_plotly(dict_list, method=method, save_path=save_path)
        fig.show()
    
    # UI 표시
    display(dropdown)
    display(output)

# 예시 사용
if __name__ == "__main__":
    # JSON 파일 경로
    json_path = "C:/Users/wjdrb/multilingual_embeddings/code-switch.json"
    
    # JSON 파일 로드
    with open(json_path, "r", encoding="UTF-8") as f:
        dict_list = json.load(f)
    
    # 특정 딕셔너리만 하이라이트하여 시각화 (예: 0번째 딕셔너리)
    # fig = visualize_embeddings_plotly(dict_list, method='pca', save_path='embeddings_dict0_highlighted', selected_dict_idx=0)
    # fig.show()
    
    # 사용자 선택 UI로 시각화
    visualize_with_dictionary_selection(dict_list, method='pca', save_path='embeddings_selected_dict')