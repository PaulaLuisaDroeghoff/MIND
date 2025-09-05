import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Your LLM training results data
data = {
    'Model': [
        'MentalManip DistilRoBERTa',
        'Iterative Feedback Agent',
        'ChatGPT-4.1 Mini Few-Shot',
        'ChatGPT-4.1 Mini Zero-Shot',
        'Random Forest',
        'Logistic Regression'
    ],
    'Precision': [0.681, 0.727, 0.615, 0.619, 0.553, 0.548],
    'Recall': [0.681, 0.640, 0.679, 0.613, 0.573, 0.576],
    'F1-Score': [0.680, 0.681, 0.646, 0.616, 0.555, 0.557]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define colors for each model
colors = [
    '#e74c3c',  # Red - MentalManip DistilRoBERTa
    '#3498db',  # Blue - Iterative Feedback Agent
    '#2ecc71',  # Green - ChatGPT Few-Shot
    '#f39c12',  # Orange - ChatGPT Zero-Shot
    '#9b59b6',  # Purple - Random Forest
    '#1abc9c'  # Teal - Logistic Regression
]

# Create the radar chart
fig = go.Figure()

# Add each model as a separate trace
for i, model in enumerate(df['Model']):
    fig.add_trace(go.Scatterpolar(
        r=[df.loc[i, 'Precision'], df.loc[i, 'Recall'], df.loc[i, 'F1-Score']],
        theta=['Precision', 'Recall', 'F1-Score'],
        fill='toself',
        fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.1)',
        line=dict(color=colors[i], width=4),  # Thicker lines
        marker=dict(
            color=colors[i],
            size=10,  # Larger markers
            line=dict(color='white', width=3)
        ),
        name=model,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      '%{theta}: %{r:.3f}<br>' +
                      '<extra></extra>'
    ))

# Update layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0.4, 0.8],  # Focus on the actual data range
            tickmode='linear',
            tick0=0.4,
            dtick=0.05,  # Smaller increments for better granularity
            tickfont=dict(size=12, color='black', family='Arial'),
            gridcolor='rgba(52, 73, 94, 0.4)',
            gridwidth=2,
            tickformat='.3f'  # Show 3 decimal places
        ),
        angularaxis=dict(
            tickfont=dict(size=16, color='black', family='Arial Black'),
            linecolor='rgba(52, 73, 94, 0.5)',
            gridcolor='rgba(52, 73, 94, 0.4)',
            gridwidth=2
        ),
        bgcolor='rgba(255, 255, 255, 0.8)'
    ),
    title=dict(
        text='<b>Sentence Level Training Results Comparison</b>',
        x=0.5,
        y=0.95,
        font=dict(size=20, color='black', family='Arial')
    ),
    font=dict(size=12, color='black'),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='rgba(52, 73, 94, 0.3)',
        borderwidth=1
    ),
    width=800,
    height=600,
    paper_bgcolor='rgba(255, 255, 255, 0.95)',
    plot_bgcolor='rgba(255, 255, 255, 0.95)'
)

# Show the plot
fig.show()

# Save the full plot as PNG
fig.write_image("sentence_radar_plot_full.png", width=800, height=600, scale=2)

# Optional: Save as HTML for interactivity
fig.write_html("sentence_radar_plot_full.html")

# Create and save the simplified top-3 plot
print("=" * 60)
print("LLM MODEL PERFORMANCE SUMMARY")
print("=" * 60)

for i, model in enumerate(df['Model']):
    precision = df.loc[i, 'Precision']
    recall = df.loc[i, 'Recall']
    f1 = df.loc[i, 'F1-Score']

    print(f"\n{model}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

# Find best performing model for each metric
print("\n" + "=" * 60)
print("BEST PERFORMERS BY METRIC")
print("=" * 60)

best_precision_idx = df['Precision'].idxmax()
best_recall_idx = df['Recall'].idxmax()
best_f1_idx = df['F1-Score'].idxmax()

print(f"Best Precision: {df.loc[best_precision_idx, 'Model']} ({df.loc[best_precision_idx, 'Precision']:.3f})")
print(f"Best Recall:    {df.loc[best_recall_idx, 'Model']} ({df.loc[best_recall_idx, 'Recall']:.3f})")
print(f"Best F1-Score:  {df.loc[best_f1_idx, 'Model']} ({df.loc[best_f1_idx, 'F1-Score']:.3f})")

# Calculate average performance
df['Average'] = df[['Precision', 'Recall', 'F1-Score']].mean(axis=1)
best_overall_idx = df['Average'].idxmax()
print(f"Best Overall:   {df.loc[best_overall_idx, 'Model']} ({df.loc[best_overall_idx, 'Average']:.3f} avg)")

# Create a simpler version with fewer models (top 3 performers)
print("\n" + "=" * 60)
print("CREATING SIMPLIFIED RADAR PLOT (TOP 3 MODELS)")
print("=" * 60)

# Get top 3 models by F1-Score
top_3_indices = df.nlargest(3, 'F1-Score').index
top_3_df = df.loc[top_3_indices]

# Create simplified radar chart
fig_simple = go.Figure()

for i, idx in enumerate(top_3_indices):
    model = df.loc[idx, 'Model']
    fig_simple.add_trace(go.Scatterpolar(
        r=[df.loc[idx, 'Precision'], df.loc[idx, 'Recall'], df.loc[idx, 'F1-Score']],
        theta=['Precision', 'Recall', 'F1-Score'],
        fill='toself',
        fillcolor=f'rgba({int(colors[idx][1:3], 16)}, {int(colors[idx][3:5], 16)}, {int(colors[idx][5:7], 16)}, 0.2)',
        line=dict(color=colors[idx], width=4),
        marker=dict(
            color=colors[idx],
            size=10,
            line=dict(color='white', width=2)
        ),
        name=model,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      '%{theta}: %{r:.3f}<br>' +
                      '<extra></extra>'
    ))

fig_simple.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0.4, 0.8],  # Focus on actual data range
            tickmode='linear',
            tick0=0.4,
            dtick=0.05,
            tickfont=dict(size=12, color='black', family='Arial'),
            gridcolor='rgba(52, 73, 94, 0.4)',
            gridwidth=2,
            tickformat='.3f'
        ),
        angularaxis=dict(
            tickfont=dict(size=16, color='black', family='Arial Black'),
            linecolor='rgba(52, 73, 94, 0.5)',
            gridcolor='rgba(52, 73, 94, 0.4)',
            gridwidth=2
        ),
        bgcolor='rgba(255, 255, 255, 0.9)'
    ),
    title=dict(
        text='<b>Top 3 LLM Models Performance</b>',
        x=0.5,
        y=0.95,
        font=dict(size=22, color='black', family='Arial')
    ),
    font=dict(size=12, color='black'),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='rgba(127, 140, 141, 0.3)',
        borderwidth=1
    ),
    width=700,
    height=700,
    paper_bgcolor='rgba(255, 255, 255, 0.95)',
    plot_bgcolor='rgba(255, 255, 255, 0.95)'
)

# Show both plots
print("Displaying full radar plot with all models...")
fig.show()

print("Displaying simplified radar plot with top 3 models...")
fig_simple.show()

# Save both plots as PNG files
print("\nSaving plots as PNG files...")
fig.write_image("sentence_radar_plot_full.png", width=800, height=600, scale=2)
fig_simple.write_image("sentence_radar_plot_top3.png", width=700, height=700, scale=2)

# Optional: Save as HTML for interactivity
fig.write_html("sentence_radar_plot_full.html")
fig_simple.write_html("sentence_radar_plot_top3.html")

print("Files saved:")
print("- sentence_radar_plot_full.png (all 6 models)")
print("- sentence_radar_plot_top3.png (top 3 models)")
print("- sentence_radar_plot_full.html (interactive version - all models)")
print("- sentence_radar_plot_top3.html (interactive version - top 3 models)")