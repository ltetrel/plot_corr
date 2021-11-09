import os
import time
import json
import pandas as pd
import numpy as np
import plotly as ply
import plotly.express
import plotly.graph_objects

def plot_feature_select(corr, p, feature_list, report_path):
  hovertext = list()
  # create hover info
  for ii in range(p.shape[0]):
    hovertext.append(list())
    for jj in range(p.shape[1]):
      if p[ii, jj] > 0.05:
        hovertext[-1].append(f"Un-correlated (p = {p[ii, jj]} > 0.05)")
      else:
        hovertext[-1].append(f"Correlated (p = {p[ii, jj]} <= 0.05)")
  # plotly figure
  # https://plotly.com/python-api-reference/plotly.graph_objects.html
  fig = ply.graph_objects.Figure(ply.graph_objects.Heatmap(
    x=feature_list
    , y=feature_list
    , z=corr
    , text=hovertext
    , hovertemplate="feature x: %{x} <br>feature y: %{y} <br>correlation: %{z} <br>%{text} <br> <extra></extra>"
    , colorscale="Viridis"))
  fig.update_layout(
    title="Spearman rank correlation (absolute value)"
    , yaxis= dict(autorange="reversed")
    , width=1300
    , height=1300)
  fig.write_html(report_path, auto_open=False)

if __name__ == "__main__":
  curr_filepath = os.path.dirname(__file__)
  plot_loss_example(os.path.join(curr_filepath, "../../reports/training/auto_regression/1632453633612"))
  plot_metrics(os.path.join(curr_filepath, "../../reports/training/auto_regression/1632453633612"))
