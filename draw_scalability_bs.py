import matplotlib.lines as mlines
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
]

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"


# GPTLarge, 32 GPUs

ddp = [
  [8,    16,    32,    60,    64,    68],  # batch size
  [1.45, 1.49,  1.59,  1.75,  2.12,  1.72],
  [7583, 11548, 19567, 33791, 35590, 37806],
]

fsdp = [
  [8,    16,   32,    60,    64,    68,    72,    76],
  [1.96, 2.77, 2.44,  2.45,  2.83,  2.63,  3.43,  3.77],
  [4470, 8423, 16436, 30679, 32459, 34686, 36467, 38692],
]

pdp_ck2 = [
  [8,    16,   32,    64,    72,    80,    88,    96],
  [1.34, 1.33, 1.47,  1.79,  1.90,  1.95,  1.97,  2.45],
  [5534, 8353, 14027, 25373, 28207, 31045, 33880, 36718],
]

pdp_ck4 = [
  [8,    16,   32,   64,    72,    80,    88,    96,    104,   112,   120,   128,   136,   144,   152,   160],
  [1.43, 1.41, 1.56, 1.84,  2.01,  2.12,  2.09,  2.26,  2.34,  2.48,  2.47,  2.67,  2.68,  2.71,  2.97,  2.99],
  [4327, 5959, 9196, 16342, 18274, 20003, 21937, 23664, 25603, 27327, 29264, 30987, 32927, 34650, 36587, 38311],
]

pdp_ck8 = [
  [144,   152,   160,   168,   176],
  [2.93,  3.03,  3.09,  3.39,  3.36],
  [25765, 26988, 28321, 29709, 31091]
]

width=3
max_qps=450

def plot_twin(model, names, bs, delay, mem, blk_size=256, show=True):
  def plot_one(model, name, ax1, bs, delay, mem, hide_y_tick, max_bs, blk_size=256):

    ws = 32
    def qps(bs, delay):
      return ws * np.array(bs) * blk_size / np.array(delay) / 1000

    def gb(mem):
      return np.array(mem) / 1000

    ax2 = ax1.twinx()
    ax1.bar(bs, gb(mem), width=width, color=COLORS[1])
    ax2.plot(bs, qps(bs, delay), '.-', color=COLORS[0])
    ax1.tick_params(axis='y', colors=COLORS[1])
    ax2.tick_params(axis='y', colors=COLORS[0])


    ax1.set_ylim([0, 40])
    ax2.set_ylim([0, max_qps])

    ax1.set_xlim([0, max_bs])
    ax2.set_xlim([0, max_bs])

    if hide_y_tick[0]:
      ax1.set_yticklabels([])

    if hide_y_tick[1]:
      ax2.set_yticklabels([])

    ax1.text(10, 35, name, size="12")
    ax1.add_patch(
      Rectangle(
        [max(bs) + 1, 0], 
        169 - max(bs), 40, 
        facecolor="#cccccc", 
        edgecolor="#cccccc",
      )
    )

    if name != "pdp4":
      ax1.text(max(bs) + 0.5 * (max_bs - max(bs)), 20, "OOM", color="w", size="15", ha="center", weight='bold')



  max_bs = np.array([max(one_bs) for one_bs in bs])
  fig = plt.figure(figsize=(12, 3))
  spec = fig.add_gridspec(
    ncols=4, nrows=1, 
    #width_ratios=max_bs / 40.0 + 1,
    hspace=0, wspace=0,
  )

  #spec.subplots(sharey=True)

  for i in range(len(bs)):
    ax = fig.add_subplot(spec[0, i])
    plot_one(model, names[i], ax, bs[i], delay[i], mem[i], [i > 0, i < 3], max(max_bs)+5, blk_size)

  handles = [
     Patch(facecolor=COLORS[1], label='Peak GPU Mem'),
     mlines.Line2D([], [], linestyle='-', marker='.', color=COLORS[0], label='QPS (1k/S)')
  ]

  fig.legend(
    handles=handles,
    labels=["Peak GPU Mem", "QPS (1k/S)"],
    bbox_to_anchor=(-0.43, 0.9, 1, 0.2),
    fancybox=True,
    ncol=2,
  )

  fig.text(0.5, -0.04, f'Batch Size ({model})', ha='center')
  fig.text(0.08, 0.5, 'Peak GPU Mem', color=COLORS[1], va='center', rotation='vertical')
  fig.text(0.945, 0.5, 'QPS (1k/Second)', color=COLORS[0], va='center', rotation='vertical')

  #plt.show()
  plt.savefig(f'image/{model}_twin_qps_max_bs.{FIG_TYPE}', bbox_inches='tight')


plot_twin(
  "GPTLarge",
  ["ddp", "fsdp", "pdp2", "pdp4"],
  [ddp[0], fsdp[0], pdp_ck2[0], pdp_ck4[0]],
  [ddp[1], fsdp[1], pdp_ck2[1], pdp_ck4[1]],
  [ddp[2], fsdp[2], pdp_ck2[2], pdp_ck4[2]],
)

"""
plot_twin("GPTLarge", ddp[0], ddp[1], ddp[2])
plot_twin("GPTLarge", fsdp[0], fsdp[1], fsdp[2])
plot_twin("GPTLarge", pdp_ck2[0], pdp_ck2[1], pdp_ck2[2])
plot_twin("GPTLarge", pdp_ck4[0], pdp_ck4[1], pdp_ck4[2])
"""

def plot_qps(model, blk_size=256, show=True):
  ws = 32
  def qps(bs, delay):
    return ws * np.array(bs) * blk_size / np.array(delay) / 1000

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], qps(ddp[0], ddp[1]), '.-', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], qps(fsdp[0], fsdp[1]), '.-', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], qps(pdp_ck2[0], pdp_ck2[1]), '.-', color=COLORS[2])[0],
    # pdp ck=4
    plt.plot(pdp_ck4[0], qps(pdp_ck4[0], pdp_ck4[1]), '^-', color=COLORS[2])[0],
    # pdp ck=8
    plt.plot(pdp_ck8[0], qps(pdp_ck8[0], pdp_ck8[1]), 's-', color=COLORS[2])[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper left",
      labels=[
        "ddp", 
        "fsdp",
        "pdp ck2",
        "pdp ck4",
        "pdp ck8",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Per-GPU Batch Size")
  plt.ylabel(f"{model} QPS (1k / Second)")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{model}_qps_max_bs.{FIG_TYPE}', bbox_inches='tight')


def plot_mem(model, blk_size=256, show=True):

  def gb(mem):
    return np.array(mem) / 1000

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], gb(ddp[2]), '.-', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], gb(fsdp[2]), '.-', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], gb(pdp_ck2[2]), '.-', color=COLORS[2])[0],
    # pdp ck=4
    plt.plot(pdp_ck4[0], gb(pdp_ck4[2]), '^-', color=COLORS[2])[0],
    # pdp ck=8
    plt.plot(pdp_ck8[0], gb(pdp_ck8[2]), 's-', color=COLORS[2])[0],
  ])

  plt.legend(
      handles=handles,
      loc="lower right",
      labels=[
        "ddp", 
        "fsdp",
        "pdp ck2",
        "pdp ck4",
        "pdp ck8",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Per-GPU Batch Size")
  plt.ylabel(f"{model} Peak GPU Memory (GB)")
  plt.ylim([0, 40])

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{model}_memory_max_bs.{FIG_TYPE}', bbox_inches='tight')


#plot_qps("GPTLarge", show=False)
#plot_mem("GPTLarge", show=False)


# GPTXXL, 32 GPUs

ddp = [
  [8,     16,    20,    24],
  [4.42,  5.62,  5.03,  5.28],
  [20258, 29088, 33737, 37932],
]

fsdp = [
  [8,    16,    28,    30,    32],
  [6.50, 7.35,  7.43,  10.61, 8.87],
  [9775, 18555, 32070, 34157, 36224],
]

pdp_ck2 = [
  [8,     16,    32,    36,    40],
  [3.92,  4.56,  4.87,  4.80,  4.72],
  [13877, 19134, 29657, 32408, 34916],
]

pdp_ck4 = [
  [8,     16,    32,    64,    72,    80],
  [3.92,  4.46,  4.07,  5.17,  5.67,  6.32],
  [11473, 14310, 19994, 31379, 34345, 37068],
]

pdp_ck8 = [
  [72,    80,    88,    96,    104],
  [6.61,  6.47,  6.80,  6.89,  7.56],
  [23366, 25126, 27012, 28896, 30837],
]


#plot_qps("GPTXXL", show=False)
#plot_mem("GPTXXL", show=False)

width=1.7
max_qps=120

plot_twin(
  "GPT2.7B",
  ["ddp", "fsdp", "pdp2", "pdp4"],
  [ddp[0], fsdp[0], pdp_ck2[0], pdp_ck4[0]],
  [ddp[1], fsdp[1], pdp_ck2[1], pdp_ck4[1]],
  [ddp[2], fsdp[2], pdp_ck2[2], pdp_ck4[2]],
)


























