import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np



COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
  "#888888",
]

HATCHES = [
  "",
  "//",
  "--",
]

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"


# world size = 32, per-GPU batch size = 16

ddp = [
  [125,  350,  760,   1300,  2700],  # GPTSmall, Medium, Large, Xl, XXL
  [0.29, 0.75, 1.43,  2.55,  5.13],  # per-iteration delay
  [3625, 8014, 11548, 16748, 29088], # Peak mem
]

ddp_checkpoint = [
  [125,  350,  760,  1300, 2700],
  [0.31, 0.85, 1.43, 2.93, 4.22],
  [2069, 3186, 5660, 9278, 17425],
]

ddp_offload = [
  [125,  350,  760,  1300, 2700],
  [0.29, 0.74, 1.39, 2.97, 4.95],
  [1994, 3047, 5649, 9278, 17427],
]

fsdp = [
  [125,  350,  760,  1300,  2700,  6700],  # GPTSmall, Large, Xl, XXL, XXXL
  [0.46, 1.24, 2.77, 3.55,  7.35,  22.89],
  [3056, 6543, 8423, 11416, 18555, 26230],
]

fsdp_checkpoint = [
  [125,  350,  760,  1300,  2700,  6700,  13000, 39000,  76000,  175000],
  [0.55, 1.45, 2.75, 5.01,  8.66,  25.85, 52.43, 143.14, 319.67, 687.32],
  [1091, 1306, 1575, 1891,  2434,  3678,  6136,  10649,  20770,  16025],
]

fsdp_offload = [
  [125,  350,  760,  1300,  2700,  6700,  13000, 39000,  76000,  175000],
  [0.54, 1.26, 2.62, 7.20,  8.63,  20.79, 60.18, 141.54, 322.60, 674.77],
  [1016, 1104, 1273, 1485,  1763,  2604,  4152,  7692,   15429,  6462],
]

# for pdp (<=6700), it's world size = 16 and batch size = 32
# for pdp (>6700), 2-device pipeline OOM, used 4-device

pdp_ck2 = [  
  [125,  350,  760,  1300,  2700,  6700],
  [0.28, 0.70, 1.38, 2.37,  4.48,  10.46],
  [3675, 6120, 8353, 11590, 19134, 34969],
]

pdp_ck2_checkpoint = [
  [125,  350,  760,  1300, 2700,  6700,  13000],
  [0.31, 0.71, 1.56, 2.92, 4.48,  10.36, 21.18],
  [3095, 3912, 5341, 7240, 11490, 24102, 29887],
]

pdp_ck2_offload = [
  [125,  350,  760,  1300, 2700,  6700,  13000],
  [0.32, 0.81, 1.52, 2.44, 4.47,  11.21, 21.65],
  [3059, 3811, 5191, 7037, 11153, 23565, 29121],
]

pdp_ck4 = [
  [125,  350,  760,  1300, 2700,  6700,  13000],  
  [0.33, 0.87, 1.41, 2.35, 4.19,  11.78, 19.76],
  [2707, 4140, 5959, 8478, 14310, 28416, 38583],
]

pdp_ck4_checkpoint = [
  [125,  350,  760,  1300, 2700,  6700,  13000, 39000], 
  [0.39, 0.95, 1.44, 2.46, 4.40,  10.70, 22.60, 36.98],
  [2257, 3036, 4434, 6304, 10484, 22982, 27474, 35091],
]

pdp_ck4_offload = [
  [125,  350,  760,  1300, 2700,  6700,  13000, 39000], 
  [0.40, 0.95, 1.67, 2.56, 4.65,  12.07, 21.17, 40.94],
  [2236, 2984, 4360, 6202, 10314, 22714, 27049, 34713],
]


width = 0.08
blk_size = 256
show=False
def plot_qps_bar_small():

  def get_qps(ws, bs, delay):
    return ws * bs * blk_size / np.array(delay) / 1000

  def plot_one_bar(ax, qps, index):
    qps = qps[:5]
    x = np.arange(len(qps)) + (index - 6) * width
    ax.bar(x, qps, width, color=COLORS[index//3], hatch=HATCHES[index%3], edgecolor="w")


  fig = plt.figure(figsize=(10, 3))

  ax = plt.gca()

  delays = [
    ddp[1],
    ddp_checkpoint[1],
    ddp_offload[1],
    fsdp[1],
    fsdp_checkpoint[1],
    fsdp_offload[1],
    pdp_ck2[1],
    pdp_ck2_checkpoint[1],
    pdp_ck2_offload[1],
    pdp_ck4[1],
    pdp_ck4_checkpoint[1],
    pdp_ck4_offload[1],
  ]

  for i, delays in enumerate(delays):
    plot_one_bar(ax, get_qps(32, 16, delays), i)

  plt.xlim([-0.6, 4.6])
  #plt.yscale("log")

  handles = [
    Patch(facecolor=COLORS[0], label='ddp'),
    Patch(facecolor=COLORS[1], label='fsdp'),
    Patch(facecolor=COLORS[2], label='pdp2'),
    Patch(facecolor=COLORS[2], label='pdp4'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="//", label='ac'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="---", label='ao'),
  ]

  plt.legend(
    handles=handles,
    labels=[
      "ddp",
      "fsdp",
      "pdp2",
      "pdp4",
      "ac",
      "ao",
    ],
    loc="upper right",
    #fancybox=True,
    ncol=3,
    prop={'size': 12},
  )

  plt.xlabel(f"Model Size")
  plt.ylabel(f"QPS (1k/Second)")

  ax.set_xticks(np.array(range(5)))
  ax.set_xticklabels(['162M','405M','834M','1.4B', '2.8B'])

  x = 4 - 0.4
  plt.text(x, 200, f'DDP OOM on\nlarger models', ha='center', color=COLORS[0], weight="bold")
  plt.arrow(x, 180, 0, -60, color=COLORS[0], width=0.01, head_length=10, head_width=0.05)

  if show:
    plt.show()
  else:
    plt.savefig(f'image/model_size_qps_small.{FIG_TYPE}', bbox_inches='tight')


def plot_mem_bar_small():

  def gb(mem):
    return np.array(mem) / 1000

  def plot_one_bar(ax, mems, index):
    mems = mems[:5]
    x = np.arange(len(mems)) + (index - 6) * width
    ax.bar(x, mems, width, color=COLORS[index//3], hatch=HATCHES[index%3], edgecolor="w")


  fig = plt.figure(figsize=(10, 3))

  ax = plt.gca()

  mems = [
    ddp[2],
    ddp_checkpoint[2],
    ddp_offload[2],
    fsdp[2],
    fsdp_checkpoint[2],
    fsdp_offload[2],
    pdp_ck2[2],
    pdp_ck2_checkpoint[2],
    pdp_ck2_offload[2],
    pdp_ck4[2],
    pdp_ck4_checkpoint[2],
    pdp_ck4_offload[2],
  ]

  for i, mems in enumerate(mems):
    plot_one_bar(ax, gb(mems), i)

  plt.xlim([-0.6, 4.6])
  #plt.yscale("log")

  handles = [
    Patch(facecolor=COLORS[0], label='ddp'),
    Patch(facecolor=COLORS[1], label='fsdp'),
    Patch(facecolor=COLORS[2], label='pdp2'),
    Patch(facecolor=COLORS[2], label='pdp4'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="//", label='ck'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="---", label='ao'),
  ]

  plt.legend(
    handles=handles,
    labels=[
      "ddp",
      "fsdp",
      "pdp2",
      "pdp4",
      "ac",
      "ao",
    ],
    loc="upper left",
    #fancybox=True,
    ncol=3,
    prop={'size': 12},
  )

  plt.xlabel(f"Model Size")
  plt.ylabel(f"Peak GPU Memory (GB)")

  ax.set_xticks(np.array(range(5)))
  ax.set_xticklabels(['162M','405M','834M','1.4B', '2.8B'])


  if show:
    plt.show()
  else:
    plt.savefig(f'image/model_size_mem_small.{FIG_TYPE}', bbox_inches='tight')



plot_qps_bar_small()
plot_mem_bar_small()



def plot_qps_bar_Large():

  def get_qps(ws, bs, delay):
    return ws * bs * blk_size / np.array(delay) / 1000

  def plot_one_bar(ax, qps, index):
    qps = qps[5:]
    x = np.arange(len(qps)) + (index - 4.5) * width
    ax.bar(x, qps, width, color=COLORS[1 + index//3], hatch=HATCHES[index%3], edgecolor="w")


  fig = plt.figure(figsize=(10, 3))

  ax = plt.gca()

  delays = [
    fsdp[1],
    fsdp_checkpoint[1],
    fsdp_offload[1],
    pdp_ck2[1],
    pdp_ck2_checkpoint[1],
    pdp_ck2_offload[1],
    pdp_ck4[1],
    pdp_ck4_checkpoint[1],
    pdp_ck4_offload[1],
  ]

  for i, delays in enumerate(delays):
    plot_one_bar(ax, get_qps(32, 16, delays), i)

  plt.xlim([-0.6, 4.6])
  #plt.yscale("log")

  handles = [
    Patch(facecolor=COLORS[1], label='fsdp'),
    Patch(facecolor=COLORS[2], label='pdp2'),
    Patch(facecolor=COLORS[2], label='pdp4'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="//", label='ck'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="---", label='ao'),
  ]

  plt.legend(
    handles=handles,
    labels=[
      "fsdp",
      "pdp2",
      "pdp4",
      "ac",
      "ao",
    ],
    loc="upper right",
    #fancybox=True,
    ncol=2,
    prop={'size': 12},
  )

  plt.xlabel(f"Model Size")
  plt.ylabel(f"QPS (1k/Second)")

  ax.set_xticks(np.array(range(5)))
  ax.set_xticklabels(['6.8B','16B','34B','81B', '175B'])


  plt.text(1, 10, f'2-device PDP OOM\nUse 4-device PDP', ha='center', color=COLORS[2], weight='bold')
  plt.arrow(1, 9.5, 0, -2, color=COLORS[2], width=0.01, head_length=0.3, head_width=0.05)

  plt.text(2.3, 7, f'4-device PDP OOM\nUse 8-device PDP', ha='center', color=COLORS[3], weight='bold')
  plt.arrow(2.3, 6.5, 0, -2, color=COLORS[3], width=0.01, head_length=0.3, head_width=0.05)

  if show:
    plt.show()
  else:
    plt.savefig(f'image/model_size_qps_large.{FIG_TYPE}', bbox_inches='tight')



def plot_mem_bar_large():

  def gb(mem):
    return np.array(mem) / 1000

  def plot_one_bar(ax, mems, index):
    mems = mems[5:]
    x = np.arange(len(mems)) + (index - 4.5) * width
    ax.bar(x, mems, width, color=COLORS[1 + index//3], hatch=HATCHES[index%3], edgecolor="w")


  fig = plt.figure(figsize=(10, 3))

  ax = plt.gca()

  mems = [
    fsdp[2],
    fsdp_checkpoint[2],
    fsdp_offload[2],
    pdp_ck2[2],
    pdp_ck2_checkpoint[2],
    pdp_ck2_offload[2],
    pdp_ck4[2],
    pdp_ck4_checkpoint[2],
    pdp_ck4_offload[2],
  ]

  for i, mems in enumerate(mems):
    plot_one_bar(ax, gb(mems), i)

  plt.xlim([-0.6, 4.6])
  #plt.yscale("log")

  handles = [
    Patch(facecolor=COLORS[1], label='fsdp'),
    Patch(facecolor=COLORS[2], label='pdp2'),
    Patch(facecolor=COLORS[2], label='pdp4'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="//", label='ck'),
    Patch(facecolor='#cccccc', edgecolor='w', hatch="---", label='ao'),
  ]

  plt.legend(
    handles=handles,
    labels=[
      "fsdp",
      "pdp2",
      "pdp4",
      "ac",
      "ao",
    ],
    loc="upper right",
    #fancybox=True,
    ncol=2,
    prop={'size': 12},
  )

  plt.xlabel(f"Model Size")
  plt.ylabel(f"Peak GPU Memory (GB)")

  ax.set_xticks(np.array(range(5)))
  ax.set_xticklabels(['6.8B','16B','34B','81B', '175B'])

  plt.text(3.8, 18, f'Enabled Parameter\nOffloading on 175B Model', ha='center', color=COLORS[1], weight='bold')
  #plt.arrow(3.8, 15, 0, -5, color=COLORS[1], width=0.01, head_length=1.5, head_width=0.05)

  if show:
    plt.show()
  else:
    plt.savefig(f'image/model_size_mem_large.{FIG_TYPE}', bbox_inches='tight')



plot_qps_bar_Large()
plot_mem_bar_large()


def plot_qps(ws, bs, blk_size=256, show=True):

  def qps(ws, bs, delay):
    return ws * bs * blk_size / np.array(delay) / 1000

  plt.figure(figsize=(8, 4))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], qps(ws, bs, ddp[1]), '.-', color=COLORS[0])[0],
    plt.plot(
        ddp_checkpoint[0], 
        qps(ws, bs, ddp_checkpoint[1]), 
        '^--', 
        color=COLORS[0]
    )[0],
    plt.plot(
        ddp_offload[0], 
        qps(ws, bs, ddp_offload[1]), 
        's:', 
        color=COLORS[0]
    )[0],
    # fsdp
    plt.plot(fsdp[0], qps(ws, bs, fsdp[1]), '.-', color=COLORS[1])[0],
    plt.plot(
        fsdp_checkpoint[0], 
        qps(ws, bs, fsdp_checkpoint[1]), 
        '^--', 
        color=COLORS[1]
    )[0],
    plt.plot(
        fsdp_offload[0], 
        qps(ws, bs, fsdp_offload[1]), 
        's:', 
        color=COLORS[1]
    )[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], qps(ws, bs, pdp_ck2[1]), '.-', color=COLORS[2])[0],
    plt.plot(
        pdp_ck2_checkpoint[0], 
        qps(ws, bs, pdp_ck2_checkpoint[1]), 
        '^--', 
        color=COLORS[2]
    )[0],
    plt.plot(   
        pdp_ck2_offload[0], 
        qps(ws, bs, pdp_ck2_offload[1]), 
        's:', 
        color=COLORS[2]
    )[0],
    # pdp ck=4
    plt.plot(pdp_ck4[0], qps(ws, bs, pdp_ck4[1]), '.-', color=COLORS[2])[0],
    plt.plot(
        pdp_ck4_checkpoint[0], 
        qps(ws, bs, pdp_ck4_checkpoint[1]), 
        '^--', 
        color=COLORS[2]
    )[0],
    plt.plot(   
        pdp_ck4_offload[0], 
        qps(ws, bs, pdp_ck4_offload[1]), 
        's:', 
        color=COLORS[2]
    )[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper right",
      labels=[
        "ddp", 
        "ddp ck",
        "ddp ol", 
        "fsdp", 
        "fsdp ck",
        "fsdp ol", 
        "pdp2", 
        "pdp2 ck",
        "pdp2 ol", 
        "pdp4", 
        "pdp4 ck",
        "pdp4 ol", 
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Model Size (Million Parameters)")
  plt.ylabel(f"QPS (1k / Second)")
  plt.xscale('log')
  plt.grid(True, which="both")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/model_qps_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


def plot_mem(bs, blk_size=256, show=True):

  def gb(mem):
    return np.array(mem) / 1000

  plt.figure(figsize=(8, 4))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], gb(ddp[2]), '.-', color=COLORS[0])[0],
    plt.plot(ddp_checkpoint[0], gb(ddp_checkpoint[2]), '^--', color=COLORS[0])[0],
    plt.plot(ddp_offload[0], gb(ddp_offload[2]), 's:', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], gb(fsdp[2]), '.-', color=COLORS[1])[0],
    plt.plot(fsdp_checkpoint[0], gb(fsdp_checkpoint[2]), '^--', color=COLORS[1])[0],
    plt.plot(fsdp_offload[0], gb(fsdp_offload[2]), 's:', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], gb(pdp_ck2[2]), '.-', color=COLORS[2])[0],
    plt.plot(pdp_ck2_checkpoint[0], gb(pdp_ck2_checkpoint[2]), '^--', color=COLORS[2])[0],
    plt.plot(pdp_ck2_offload[0], gb(pdp_ck2_offload[2]), 's:', color=COLORS[2])[0],
    # pdp ck=4
    plt.plot(pdp_ck4[0], gb(pdp_ck4[2]), '.-', color=COLORS[3])[0],
    plt.plot(pdp_ck4_checkpoint[0], gb(pdp_ck4_checkpoint[2]), '^--', color=COLORS[3])[0],
    plt.plot(pdp_ck4_offload[0], gb(pdp_ck4_offload[2]), 's:', color=COLORS[3])[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper left",
      labels=[
        "ddp", 
        "ddp ck",
        "ddp ol", 
        "fsdp", 
        "fsdp ck",
        "fsdp ol", 
        "pdp2", 
        "pdp2 ck",
        "pdp2 ol", 
        "pdp4", 
        "pdp4 ck",
        "pdp4 ol", 
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Model Size (Million Parameters)")
  plt.ylabel(f"Peak GPU Memory (GB)")
  plt.ylim([0, 40])
  plt.xscale('log')
  plt.grid(True, which="both")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/model_memory_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


#plot_qps(32, 16, show=False)
#plot_mem(16, show=False)



























