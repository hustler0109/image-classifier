from turtle import width
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("docs/model_logs.csv")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=df["epoch"] + 1, y=df["val_loss"], name="Validation Loss"),
)
fig.add_trace(
    go.Scatter(x=df["epoch"] + 1, y=df["train_loss"], name="Training Loss"),
)
fig.add_trace(
    go.Scatter(x=df["epoch"] + 1, y=df["val_acc"], name="Validation Accuracy"),
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(x=df["epoch"] + 1, y=df["train_acc"], name="Training Accuracy"),
    secondary_y=True,
)
fig.update_layout(title="Model Logs", xaxis_title="Epoch", yaxis_title="Loss/Accuracy")

fig.write_image("docs/model_logs.png", width=800, height=600)
