import plotly.express as px
import pandas as pd
import json
from pathlib import Path

json_p: Path = Path(__file__).resolve().parent.parent / "Distilled_data" / "Final.json"
with open(json_p, 'r', encoding='utf-8') as f:
    data = json.load(f)


df = pd.DataFrame(data)

fig = px.scatter_3d(
    df,
    x='valence',
    y='arousal',
    z='dominance',
    color='term',
    hover_name='term', 
    title='VAD (Valence, Arousal, Dominance) 3D Vector Space',
    labels={
        'valence': 'Valence (Pleasure/Displeasure)',
        'arousal': 'Arousal (Activation/Deactivation)',
        'dominance': 'Dominance (Control/Lack of Control)'
    }
)

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-1, 1], autorange=False, showgrid=True, gridwidth=1, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(range=[-1, 1], autorange=False, showgrid=True, gridwidth=1, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        zaxis=dict(range=[-1, 1], autorange=False, showgrid=True, gridwidth=1, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=50) 
)

fig.show()

#fig.write_html("vad_3d_plot.html")