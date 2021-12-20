import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np



COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
]

HATCHS = {
  "": None,
  "no": None,
  "cp": "//",
  "ol": "\\\\",
}

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"

INF = float("inf")

# 8 GPUs in total, 4 Machines, 2GPU per machine

ddp = [
  [125,   350,   760,   1300,  2700,  6700,  13000, 39000,  76000],  # model size
  [0.26,  0.73,  1.50,  2.18,  4.31,  INF,  INF,   INF,    INF],    # bs=8
  [2195,  4872,  7581,  11355, 20267, INF,  INF,   INF,    INF],    # mem
  ["no",  "no",  "no",  "no",  "no",  "",    "",    "",     ""],     # config
  [0.26,  0.72,  1.35,  2.77,  4.45,  INF,   INF,   INF,    INF],    # bs=16
  [3625,  8014,  11554, 16748, 17435, INF,   INF,   INF,    INF],
  ["no",  "no",  "no",  "no",  "cp",  "",    "",    "",     ""],     
  [0.30,  0.90,  1.47,  2.98,  4.63,  INF,   INF,   INF,    INF],    # bs=32
  [6506,  14031, 19567, 27592, 18516, INF,   INF,   INF,    INF],
  ["no",  "no",  "no",  "no",  "cp",  "",    "",    "",     ""],     
  [0.35,  0.81,  1.44,  3.47,  5.40,  INF,   INF,   INF,    INF],    # bs=64
  [12265, 26872, 35590, 12723, 20677, INF,   INF,   INF,    INF],
  ["no",  "no",  "no",  "cp",  "cp",  "",    "",    "",     ""],  
  [0.44,  1.06,  1.78,  3.77,  6.01,  INF,   INF,   INF,    INF],    # bs=128
  [23789, 13406, 16142, 19505, 27214, INF,   INF,   INF,    INF],
  ["no",  "cp",  "cp",  "cp",  "cp",  "",    "",    "",     ""],  
  [0.78,  1.91,  3.07,  6.71,  INF,   INF,   INF,   INF,    INF],    # bs=256
  [21964, 25079, 28739, 33050, INF,   INF,   INF,   INF,    INF],
  ["cp",  "cp",  "cp",  "cp",  "",    "",    "",    "",     ""],  
]

fsdp = [
  [125,   350,   760,   1300,  2700,  6700,  13000, 39000,  76000],  # model size
  [0.45,  1.19,  2.14,  4.04,  7.01,  16.10, 0.00,  0.00,   0.00],   # bs=8
  [1676,  3570,  4768,  6510,  10772, 16673, 0000,  0000,   0000],   # mem
  ["no",  "no",  "no",  "no",  "no",  "no",  "cp",  "ol",   "ol"],     # config
  [0.46,  1.16,  2.01,  4.25,  7.05,  19.13, 0.00,  0.00,   0.00],   # bs=16
  [3109,  6711,  8721,  11908, 19546, 28764, 0000,  0000,   0000],
  ["no",  "no",  "no",  "no",  "no",  "no",  "cp",  "ol",   "ol"], 
  [0.49,  1.31,  2.18,  4.26,  8.96,  25.49, 0.00,  0.00,   0.00],   # bs=32
  [5990,  12999, 16735, 22756, 37219, 8535,  0000,  0000,   0000],
  ["no",  "no",  "no",  "no",  "no",  "cp",  "ol",  "ol",   "ol"], 
  [0.52,  1.24,  2.26,  5.78,  10.69, 24.97, 0.00,  0.00,   0.00],   # bs=64
  [11751, 25569  23757, 6244,  8414,  9219,  0000,  0000,   0000],
  ["no",  "no",  "no",  "cp",  "cp",  "ol",  "ol",  "ol",   "ol"], 
  [0.61,  INF,   3.49,  6.26,  10.78, 26.05, 0.00,  0.00,   0.00],   # bs=128
  [23275, INF,   10032, 11432, 10138, 14750, 0000,  0000,   0000],
  ["no",  "",    "cp",  "cp",  "cp",  "ol",  "ol",  "ol",   "ol"], 
  [0.97,  INF,   4.07,  9.23,  13.62, 41.06, 0.00,  0.00,   0.00],   # bs=256
  [14896, INF,   19352, 22068, 29185, 25809, 0000,  0000,   0000],
  ["cp",  "",    "cp",  "cp",  "cp",  "ol",  "ol",  "ol",   "ol"], 
]


# pipeline length = 2 since there are only two GPUs per machine
pdp = [
  [125,   350,   760,   1300,  2700,   6700,   13000, 39000,  76000],  # model size
  [0.23,  0.61,  1.06,  1.86,  3.43,   9.75,   INF,   INF,    INF],    # bs=8 (pdp bs=16)
  [2131,  3722,  5534,  8052,  13877,  27972,  INF,   INF,    INF],    # mem
  ["no",  "no",  "no",  "no",  "no",   "no",   "",    "",     ""],     # config
  [0.25,  0.60,  1.17,  2.35,  3.68,   8.61,   INF,   INF,    INF],    # bs=16 (pdp bs=32)
  [3675,  6120,  8353,  11590, 19134,  34969,  INF,   INF,    INF],
  ["no",  "no",  "no",  "no",  "no",   "no",   "",    "",     ""],
  [0.29,  0.72,  1.35,  2.42,  4.22,   9.83,   INF,   INF,    INF],    # bs=32 (pdp bs=64)
  [6763,  10920, 14027, 18684, 29657,  35854,  INF,   INF,    INF],
  ["no",  "no",  "no",  "no",  "no",   "no4",  "",    "",     ""],
  [0.41,  0.96,  1.68,  3.26,  5.37,   12.21,  INF,   INF,    INF],    # bs=64 (pdp bs=128)
  [12948, 20522, 25373, 32827, 31379,  37628,  INF,   INF,    INF],
  ["no",  "no",  "no",  "no",  "no4",  "no8",  "",    "",     ""],
  [0.61,  1.46,  2.48,  5.00,  7.54,   19.70,  INF,   INF,    INF],    # bs=128 (pdp bs=256)
  [19282, 26600, 30987, 37794, 36666,  30305,  INF,   INF,    INF],
  ["no",  "no4", "no4", "no4", "no8",  "cp16", "",    "",     ""],
  [1.05,  2.91,  4.81,  10.04, 14.49,  INF,    INF,   INF,    INF],    # bs=256 (pdp bs=512)
  [38233, 31444, 32895, 34665, 33999,  INF,    INF,   INF,    INF], 
  ["no",  "cp4", "cp4", "cp4", "cp16", "",     "",    "",     ""],  
]



def plot_network(nnode, show=True):
  plt.figure(figsize=(6, 3))

  num_bs = len(ddp) // 3
  num_model = len(ddp[0])

  plt.xlim([0, num_model])
  plt.ylim([0, num_bs])

  ax = plt.gca()
  for bs in range(num_bs):
    for m in range(num_model):
      col = m
      row = (bs * 3) + 1

      delays = [ddp[row][col], fsdp[row][col], pdp[row][col]]
      index = delays.index(min(delays))
      color = COLORS[index]
      hatch_strs = [ddp[row + 2][col], fsdp[row + 2][col], pdp[row + 2][col]]
      hatch = HATCHS[hatch_strs[index][:2]]


      ax.add_patch(
        Rectangle(
          [m + 0.05, bs +0.05], 
          0.95, 0.95, 
          facecolor=color, 
          edgecolor="w",
          hatch=hatch,
        )
      )

  plt.xlabel(f"Model Size")
  plt.ylabel(f"Batch Size")

  ax.set_xticks(np.array(range(9)) + 0.5)
  ax.set_xticklabels(['125M','350M','760M','1.3B', '2.7B', '6.7B', '13B', '37B', '76B'])


  ax.set_yticks(np.array(range(6)) + 0.5)
  ax.set_yticklabels(['8', '16', '32', '64', '128', '256'])


  handles = [
    Patch(facecolor=COLORS[0], label='ddp'),
    Patch(facecolor=COLORS[1], label='fsdp'),
    Patch(facecolor=COLORS[2], label='pdp'),
    Patch(facecolor='grey', edgecolor='w', hatch="//", label='ck'),
    Patch(facecolor='grey', edgecolor='w', hatch="\\\\", label='ol'),
  ]

  plt.legend( 
    handles=handles,
    loc="upper left",
    prop={'family':FONT['fontname'], 'size':FONT['size']},
    bbox_to_anchor=(-0.04, 1.02, 1, 0.2),
    fancybox=True,
    ncol=5,
  )

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{nnode}node_ws8_best_strategy.{FIG_TYPE}', bbox_inches='tight')


plot_network(nnode=4, show=False)











































