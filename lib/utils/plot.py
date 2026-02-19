import matplotlib.pyplot as plt


def update_rc_params():
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams.update(
        {
            "figure.figsize": (2, 2),  # 单栏图宽度约8.8cm，转为英寸约3.5英寸
            "figure.dpi": 300,  # 高分辨率
            "font.family": "Arial",  # Nature要求的字体
            "font.size": 6,  # 字体大小
            "axes.linewidth": 0.5,  # 轴线宽度
            "axes.labelsize": 6,  # 轴标签字体大小
            'axes.labelpad': 1.417325,  # 轴标签与轴线的距离，0.5毫米
            "xtick.labelsize": 6,  # x轴刻度标签字体大小
            "ytick.labelsize": 6,  # y轴刻度标签字体大小
            "xtick.major.width": 0.5,  # x轴主刻度线宽度
            "ytick.major.width": 0.5,  # y轴主刻度线宽度
            "ytick.major.size": 1.417325,  # y轴主刻度线长度，0.5毫米
            "xtick.major.size": 1.417325,  # x轴主刻度线长度，0.5毫米
            "xtick.minor.size": 1.417325 / 2,  # x轴刻度标签与刻度线之间的距离，0.25毫米
            "ytick.minor.size": 1.417325 / 2,  # y轴刻度标签与刻度线之间的距离，0.25毫米
            # "xtick.major.pad": 1.417325,  # x轴刻度标签与刻度线之间的距离，0.5毫米
            # "ytick.major.pad": 1.417325,  # y轴刻度标签与刻度线之间的距离，0.5毫米
            "xtick.direction": "out",  # x轴刻度线朝内
            "ytick.direction": "out",  # y轴刻度线朝内
            "lines.linewidth": 0.5,  # 线宽
            "lines.markersize": 3,  # 标记大小
            "legend.fontsize": 6,  # 图例字体大小
            "legend.frameon": False,  # 不显示图例边框
            "savefig.bbox": "tight",  # 保存图片时裁剪空白边缘
            "savefig.pad_inches": 0.01,  # 保存图片时的边距
            # 'axes.spines.right': False,    # 不显示右边框
            # 'axes.spines.top': False,       # 不显示上边框
        }
    )
