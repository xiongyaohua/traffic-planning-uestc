import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np


network_df = pd.read_csv('networks/SiouxFalls_net.csv', sep='\t')


G = nx.DiGraph()


for _, row in network_df.iterrows():
    G.add_edge(
        row['init_node'], 
        row['term_node'], 
        capacity=round(row['capacity'], 2), 
        length=row['length'], 
        free_flow_time=row['free_flow_time'],
        b=row['b'],
        power=row['power'],
        speed=row['speed'],
        toll=row['toll'],
        link_type=row['link_type']
    )

pos = nx.spring_layout(G, seed=42)

edge_x = []
edge_y = []
annotations = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    edge_info = (
        f"Capacity: {edge[2]['capacity']}, "
        f"Length: {edge[2]['length']}, "
        f"Free_flow_time:{edge[2]['free_flow_time']}"
    )

    annotations.append(
        dict(
            x=mid_x, y=mid_y, xref="x", yref="y",
            text=edge_info, showarrow=False,
            font=dict(size=10)
        )
    )

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    mode='lines')


node_x = []
node_y = []
node_ids = list(G.nodes())  
colors = np.random.randint(0, 255, size=(len(node_ids), 3))  
node_color = ['rgb({},{},{})'.format(*color) for color in colors]  

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    marker=dict(
        showscale=False,  
        color=node_color,  
        size=10,
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>交通分配算法实现',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=annotations,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


fig.write_html("sioux_falls_network_unique_node_colors.html")
print("Network visualization saved to sioux_falls_network_unique_node_colors.html")