import matplotlib.pyplot as plt
from pandas.plotting import table
import numpy as np
import six


def create_table(data_farme, name, columns):

    ax = plt.subplot(111, frame_on=False)
    render_table(data_farme, columns)

    plt.savefig("./results/" + name)


def render_table(data, columns, col_width=10.0, row_height=1, font_size=10,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, colWidths=[x for x in columns], **kwargs)

    mpl_table.auto_set_font_size(True)
    mpl_table.set_fontsize(font_size)
    # mpl_table.auto_set_column_width(col=list(range(len(data.columns))))

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax
