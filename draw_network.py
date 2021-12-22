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
  "po": "--"
}

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"

INF = float("inf")

# 8 GPUs in total, 4 Machines, 2GPU per machine

ddp = [
  [125,   350,   760,   1300,  2700,  6700,  13000, 39000,  76000, 100000],  # model size
  [0.26,  0.73,  1.50,  2.18,  4.31,  INF,   INF,   INF,    INF,   INF],    # bs=8
  [2195,  4872,  7581,  11355, 20267, INF,   INF,   INF,    INF,   INF],    # mem
  ["no",  "no",  "no",  "no",  "no",  "",    "",    "",     "",    ""],     # config
  [0.26,  0.72,  1.35,  2.77,  4.45,  INF,   INF,   INF,    INF,   INF],    # bs=16
  [3625,  8014,  11554, 16748, 17435, INF,   INF,   INF,    INF,   INF],
  ["no",  "no",  "no",  "no",  "cp",  "",    "",    "",     "",    ""],     
  [0.30,  0.90,  1.47,  2.98,  4.63,  INF,   INF,   INF,    INF,   INF],    # bs=32
  [6506,  14031, 19567, 27592, 18516, INF,   INF,   INF,    INF,   INF],
  ["no",  "no",  "no",  "no",  "cp",  "",    "",    "",     "",    ""],     
  [0.35,  0.81,  1.44,  3.47,  5.40,  INF,   INF,   INF,    INF,   INF],    # bs=64
  [12265, 26872, 35590, 12723, 20677, INF,   INF,   INF,    INF,   INF],
  ["no",  "no",  "no",  "cp",  "cp",  "",    "",    "",     "",    ""],  
  [0.44,  1.06,  1.78,  3.77,  6.01,  INF,   INF,   INF,    INF,   INF],    # bs=128
  [23789, 13406, 16142, 19505, 27214, INF,   INF,   INF,    INF,   INF],
  ["no",  "cp",  "cp",  "cp",  "cp",  "",    "",    "",     "",    ""],  
  [0.78,  1.91,  3.07,  6.71,  INF,   INF,   INF,   INF,    INF,   INF],    # bs=256
  [21964, 25079, 28739, 33050, INF,   INF,   INF,   INF,    INF,   INF],
  ["cp",  "cp",  "cp",  "cp",  "",    "",    "",    "",     "",    ""],  
]

# for 76B model, FSDP wraps Linear, all other experiments, FSDP wraps transformer

fsdp = [
  [125,   350,   760,   1300,  2700,  6700,  13000,  39000,  76000,  100000],    # model size
  [0.45,  1.19,  2.14,  4.04,  7.01,  16.10, 44.02,  116.36, 313.69, 409.49],    # bs=8
  [1676,  3570,  4768,  6510,  10772, 16673, 33291,  21383,  3736,   3738],      # mem
  ["no",  "no",  "no",  "no",  "no",  "no",  "no",   "cp",   "olpo", "olpo"],    # config
  [0.46,  1.16,  2.01,  4.25,  7.05,  19.13, 52.83,  112.48, 297.81, 411.16],    # bs=16
  [3109,  6711,  8721,  11908, 19546, 28764, 12191,  20678,  4948,   4950],
  ["no",  "no",  "no",  "no",  "no",  "no",  "cp",   "ol",   "olpo", "olpo"], 
  [0.49,  1.31,  2.18,  4.26,  8.96,  25.49, 56.25,  118.36, 350.4,  392.73],    # bs=32
  [5990,  12999, 16735, 22756, 37219, 8535,  15880,  22777,  7372,   7374],
  ["no",  "no",  "no",  "no",  "no",  "cp",  "cp",   "ol",   "olpo", "olpo"], 
  [0.52,  1.24,  2.26,  5.78,  10.69, 24.97, 60.64,  139.11, 345.52, 535.76],    # bs=64
  [11751, 25569, 23757, 6244,  8414,  9219,  15238,  27287,  7372,   13058],
  ["no",  "no",  "no",  "cp",  "cp",  "ol",  "ol",   "ol",   "olpo", "olpo"], 
  [0.61,  1.72,  3.49,  6.26,  10.78, 26.05, 72.58,  182.90, 412.56, 573.75],    # bs=128
  [23275, 8823,  10032, 11432, 10138, 14750, 21957,  36308,  24426,  24428],
  ["no",  "cp",  "cp",  "cp",  "cp",  "ol",  "ol",   "ol",   "olpo", "olpo"], 
  [0.97,  2.45,  4.07,  9.23,  13.62, 41.06, 86.33,  INF,    INF,    INF],       # bs=256
  [14896, 17219, 19352, 22068, 29185, 25809, 27326,  INF,    INF,    INF],
  ["cp",  "cp",  "cp",  "cp",  "cp",  "ol",  "olpo", "",     "",     ""],        # 13B+ wraps linear
]


# pipeline length = 2 since there are only two GPUs per machine
pdp = [
  [125,   350,   760,   1300,  2700,   6700,   13000, 39000,  76000, 100000],  # model size
  [0.23,  0.61,  1.06,  1.86,  3.43,   9.75,   INF,   INF,    INF,   INF],     # bs=8 (pdp bs=16)
  [2131,  3722,  5534,  8052,  13877,  27972,  INF,   INF,    INF,   INF],     # mem
  ["no",  "no",  "no",  "no",  "no",   "no",   "",    "",     "",    ""],      # config
  [0.25,  0.60,  1.17,  2.35,  3.68,   8.61,   INF,   INF,    INF,   INF],     # bs=16 (pdp bs=32)
  [3675,  6120,  8353,  11590, 19134,  34969,  INF,   INF,    INF,   INF],
  ["no",  "no",  "no",  "no",  "no",   "no",   "",    "",     "",    ""],
  [0.29,  0.72,  1.35,  2.42,  4.22,   9.83,   INF,   INF,    INF,   INF],     # bs=32 (pdp bs=64)
  [6763,  10920, 14027, 18684, 29657,  35854,  INF,   INF,    INF,   INF],
  ["no",  "no",  "no",  "no",  "no",   "no4",  "",    "",     "",    ""],
  [0.41,  0.96,  1.68,  3.26,  5.37,   12.21,  INF,   INF,    INF,   INF],     # bs=64 (pdp bs=128)
  [12948, 20522, 25373, 32827, 31379,  37628,  INF,   INF,    INF,   INF],
  ["no",  "no",  "no",  "no",  "no4",  "no8",  "",    "",     "",    ""],
  [0.61,  1.46,  2.48,  5.00,  7.54,   19.70,  INF,   INF,    INF,   INF],     # bs=128 (pdp bs=256)
  [19282, 26600, 30987, 37794, 36666,  30305,  INF,   INF,    INF,   INF],
  ["no",  "no4", "no4", "no4", "no8",  "cp16", "",    "",     "",    ""],
  [1.05,  2.91,  4.81,  10.04, 14.49,  INF,    INF,   INF,    INF,   INF],     # bs=256 (pdp bs=512)
  [38233, 31444, 32895, 34665, 33999,  INF,    INF,   INF,    INF,   INF], 
  ["no",  "cp4", "cp4", "cp4", "cp16", "",     "",    "",     "",    ""],  
]


# add the following test with wrap=linear and cpu_offload=True
# srun --label launch_cluster_fsdp.sh GPT76B 8 offload



def plot_winner(nnode, show=True):
  plt.figure(figsize=(6.2, 3))

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
      color = 'w' if delays[index] == INF else COLORS[index]
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

      if hatch_strs[index][-2:] == "po":
        ax.add_patch(
          Rectangle(
            [m + 0.05, bs +0.05], 
            0.95, 0.95, 
            facecolor=(1, 1, 1, 0), 
            edgecolor=(1, 1, 1, 1),
            hatch=HATCHS["po"],
          )
        )


  plt.xlabel(f"Model Size")
  plt.ylabel(f"Batch Size")

  ax.set_xticks(np.array(range(10)) + 0.5)
  ax.set_xticklabels(['125M','350M','760M','1.3B', '2.7B', '6.7B', '13B', '37B', '76B', '100B'])


  ax.set_yticks(np.array(range(6)) + 0.5)
  ax.set_yticklabels(['8', '16', '32', '64', '128', '256'])


  handles = [
    Patch(facecolor=COLORS[0], label='ddp'),
    Patch(facecolor=COLORS[1], label='fsdp'),
    Patch(facecolor=COLORS[2], label='pdp'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="//", label='ck'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="\\\\", label='ol'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="--", label='po'),
  ]

  plt.legend( 
    handles=handles,
    loc="upper left",
    #prop={'family':FONT['fontname'], 'size':FONT['size']},
    bbox_to_anchor=(-0.04, 1.02, 1, 0.2),
    fancybox=True,
    ncol=6,
  )

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{nnode}node_ws8_best_strategy.{FIG_TYPE}', bbox_inches='tight')


plot_winner(nnode=4, show=False)


ddp_4n = ddp
ddp = [
  [125,   350,   760,   1300,  2700,  6700,  13000, 39000,  76000,  100000],  # model size
  [0.07,  0.12,  0.12,  0.20,  0.29,  INF,   INF,   INF,    INF,    INF],    # bs=8
  [2192,  4872,  7558,  11350, 20243, INF,   INF,   INF,    INF,    INF],    # mem
  ["no",  "no",  "no",  "no",  "no",  "",    "",    "",     "",     ""],     # config
  [0.08,  0.15,  0.20,  0.38,  0.51,  INF,   INF,   INF,    INF,    INF],    # bs=16
  [3625,  8013,  11514, 16749, 29072, INF,   INF,   INF,    INF,    INF],
  ["no",  "no",  "no",  "no",  "no",  "",    "",    "",     "",     ""],
  [0.11,  0.23,  0.34,  0.67,  1.19,  INF,   INF,   INF,    INF,    INF],    # bs=32
  [6499,  14301, 19527, 27598, 18490, INF,   INF,   INF,    INF,    INF],
  ["no",  "no",  "no",  "no",  "cp",  "",    "",    "",     "",     ""],
  [0.17,  0.39,  0.59,  1.71,  2.21,  INF,   INF,   INF,    INF,    INF],    # bs=64
  [12259, 26871, 35556, 12730, 20652, INF,   INF,   INF,    INF,    INF],
  ["no",  "no",  "no",  "no",  "cp",  "",    "",    "",     "",     ""],
  [0.30,  0.95,  1.51,  3.29,  4.25,  INF,   INF,   INF,    INF,    INF],    # bs=128
  [23783, 13405, 16099, 19509, 27190, INF,   INF,   INF,    INF,    INF],
  ["no",  "cp",  "cp",  "cp",  "cp",  "",    "",    "",     "",     ""],
  [0.72,  1.83,  2.93,  6.49,  INF,   INF,   INF,   INF,    INF,    INF],    # bs=256
  [21958, 25079, 28695, 33050, INF,   INF,   INF,   INF,    INF,    INF],
  ["cp",  "cp",  "cp",  "cp",  "",    "",    "",    "",     "",     ""], 
]


# for 76B model, FSDP wraps Linear, all other experiments, FSDP wraps transformer
fsdp_4n = fsdp
fsdp = [
  [125,   350,   760,   1300,  2700,  6700,   13000,  39000,  76000,  100000],  # model size
  [0.16,  0.28,  0.28,  0.31,  0.41,  0.71,   9.97,   21.40,  75.89,  94.83],  # bs=8
  [1676,  3554,  4768,  6509,  10775, 16673,  33302,   19872,  3736,  3738], 
  ["no",  "no",  "no",  "no",  "no",  "no",   "nopo", "olpo", "olpo", "olpo"],
  [0.17,  0.28,  0.28,  0.43,  0.58,  1.42,   8.81,   26.01,  75.88,  95.61],  # bs=16
  [3109,  6696,  8721,  11907, 19547, 6212,   12182,  20679,  4948,   4950], 
  ["no",  "no",  "no",  "no",  "no",  "cp",   "cppo", "olpo", "olpo", "olpo"], 
  [0.15,  0.28,  0.38,  0.72,  1.27,  2.47,   12.36,  34.76,  84.60,  104.63],  # bs=32
  [5990,  12984, 16735, 22756, 5001,  8533,   15870,  22777,  7372,   7374], 
  ["no",  "no",  "no",  "no",  "cp",  "cp",   "cppo", "olpo", "olpo", "olpo"],
  [0.19,  0.43,  0.63,  1.76,  2.27,  4.50,   21.48,  50.27,  108.14, 134.40], # bs=64
  [11751, 25566, 32757, 6244,  8414,  13380,  15227,  27286,  13055,  13058], 
  ["no",  "no",  "no",  "cp",  "cp",  "cp",   "olpo", "olpo", "olpo", "olpo"], 
  [0.32,  0.98,  1.53,  3.33,  4.30,  11.24,  38.66,  97.71,  159.68, INF], # bs=128
  [23275, 8823,  10032, 11432, 15339, 14750,  21958,  36308,  24426,  INF],
  ["no",  "cp",  "cp",  "cp",  "cp",  "ol",   "olpo", "olpo", "olpo", "olpo"],
  [0.72,  1.85,  2.92,  6.52,  9.72,  31.33,  INF,    INF,    INF,    INF],    # bs = 256
  [14896, 17219, 19352, 22069, 29185, 25809,  INF,    INF,    INF,    INF],
  ["no",  "cp",  "cp",  "cp",  "cp",  "olpo", "",     "",     "",     ""],
]


pdp_4n = pdp
pdp = [
  [125,   350,   760,   1300,   2700,   6700,   13000,  39000,  76000, 100000],  # model size
  [0.11,  0.19,  0.20,  0.32,   0.44,   0.86,   INF,    INF,    INF,    INF], # bs=8 (pdp bs=16)
  [2131,  3721,  5518,  8051,   13858,  27973,  INF,    INF,    INF,    INF],
  ["no",  "no",  "no",  "no",   "no",   "no",   "",     "",     "",     ""],
  [0.12,  0.23,  0.32,  0.61,   0.79,   1.46,   INF,    INF,    INF,    INF],  # bs=16 (pdp bs=32)
  [3679,  6120,  8330,  11588,  19114,  28417,  INF,    INF,    INF,    INF],
  ["no",  "no",  "no",  "no",   "no",   "no4",  "",     "",     "",     ""],
  [0.17,  0.36,  0.55,  1.12,   1.37,   2.66,   INF,    INF,    INF,    INF],  # bs=32 (pdp bs=64)
  [6761,  10920, 14004, 18686,  15150,  23869,  INF,    INF,    INF,    INF],
  ["no",  "no",  "no",  "no",   "no8",  "cp8",  "",     "",     "",     ""],
  [0.29,  0.66,  1.01,  2.11,   2.53,   4.98,   INF,    INF,    INF,    INF], # bs=64 (pdp bs=128)
  [12945, 20522, 25350, 32828,  21698,  37628,  INF,    INF,    INF,    INF],
  ["no",  "no",  "no",  "no",   "no8",  "no8",  "",     "",     "",     ""],
  [0.53,  1.16,  1.79,  3.73,   4.85,   12.09,  INF,    INF,    INF,    INF], # bs=128 (pdp bs=256)
  [25313, 26600, 22974, 27048,  36646,  30305,  INF,    INF,    INF,    INF],
  ["no",  "no",  "no8", "no8",  "no8",  "cp16", "",     "",     "",     ""],
  [0.97,  2.82,  4.49,  9.27,   11.96,  INF,    INF,    INF,    INF,    INF],  # bs=256 (pdp bs=512)
  [38230, 31444, 32873, 30884,  33980,  INF,    INF,    INF,    INF,    INF],
  ["no4", "cp4", "cp4", "cp16", "cp16", "",     "",     "",     "",     ""],
]


plot_winner(nnode=1, show=False)



def plot_network(nnode, bs_row, show=True):
  def get_qps_ratio(local, dist):
    ratio = []
    for l, d in zip(local, dist):
      if l != INF and d != INF:
        ratio.append(d / l)
      else:
        ratio.append(0)

    return ratio

  ddp_ratio = get_qps_ratio(ddp[bs_row], ddp_4n[bs_row])
  fsdp_ratio = get_qps_ratio(fsdp[bs_row], fsdp_4n[bs_row])
  pdp_ratio = get_qps_ratio(pdp[bs_row], pdp_4n[bs_row])


  plt.figure(figsize=(6, 3))
  plt.ylim([0, 25])

  width = 0.25
  x = np.arange(len(ddp_ratio))
  plt.bar(x - width, ddp_ratio, width, color=COLORS[0], label='ddp')
  plt.bar(x, pdp_ratio, width, color=COLORS[2], label='pdp')
  plt.bar(x + width, fsdp_ratio, width, color=COLORS[1], label='fsdp')

  plt.xlabel(f"Model Size")
  plt.ylabel(f"Faster Network Speedup Radio")
  plt.grid(True, which="both")

  ax = plt.gca()
  ax.set_xticks(np.array(range(10)))
  ax.set_xticklabels(['125M','350M','760M','1.3B', '2.7B', '6.7B', '13B', '37B', '76B', '100B'])


  plt.legend(    
    ["ddp", "pdp", "fsdp"],
    loc="upper right",
    prop={'family':FONT['fontname'], 'size':FONT['size']},
    fancybox=True,
  )


  if show:
    plt.show()
  else:
    plt.savefig(f'image/{nnode}node_network_speedup_bsw{bs_row}.{FIG_TYPE}', bbox_inches='tight')



plot_network(nnode=4, bs_row=1, show=False)
plot_network(nnode=4, bs_row=10, show=False)




































