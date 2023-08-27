import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from io import BytesIO

plt.rc('font', size=22)

COLOUR_1, COLOUR_2, COLOUR_LINE = "#2dd4bf", "#3b82f6", "#155e75"
LEGEND_SIZE = 16

def bar_from(x_data, y_data1, y_data2, t1=None, t2=None):
    plt.clf()
    plt.figure(figsize=(10, 8))
    x = np.arange(len(x_data))

    bar_width = 0.35
    #bar_positions = x - (bar_width * (len(x_data) - 1) / 2)
    font_props = {'weight': 'bold'}

    plt.bar(x - bar_width/2, y_data1, width=bar_width, label='Top Fitness', color=COLOUR_1)
    plt.bar(x + bar_width/2, y_data2, width=bar_width, label='Avg Fitness', color=COLOUR_2)
    plt.xticks(x, x_data)
    plt.xticks(fontproperties=font_props)
    plt.yticks(fontproperties=font_props)

    plt.axhline(y=t1, color=COLOUR_LINE, linestyle='-', linewidth=2)
    plt.axhline(y=t2, color=COLOUR_LINE, linestyle='-', linewidth=2)

    plt.subplots_adjust(top=1, bottom=0.2)
    plt.xlabel("Generation", fontsize=20, fontweight='bold')
    plt.ylabel("Fitness", fontsize=20, fontweight='bold') 

    legend = plt.legend()
    legend.get_texts()[0].set_fontweight('bold')
    legend.get_texts()[1].set_fontweight('bold')
    legend.get_texts()[0].set_fontsize(LEGEND_SIZE)
    legend.get_texts()[1].set_fontsize(LEGEND_SIZE)

    legend.get_patches()[0].set_height(LEGEND_SIZE/2)
    legend.get_patches()[1].set_height(LEGEND_SIZE/2)  

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer
