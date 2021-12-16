import matplotlib.pyplot as plt
import numpy as np



COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
]

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"

# GPTSmall

ddp = [
  [8,    16,   32,   64,   128],  # number of GPUs
  [0.08, 0.23, 0.31, 0.30, 0.32],  # delay bs = 8
  [2195, 2195, 2195, 2195, 2195],  # memory
  [0.09, 0.23, 0.31, 0.31, 0.36],  # delay bs = 20
  [4355, 4355, 4355, 4355, 4355],  # memory
]

fsdp = [
  [8,    16,   32,   64,   128], 
  [0.15, 0.40, 0.47, 0.58, 0.75],
  [1676, 1642, 1625, 1616, 1613],  
  [0.18, 0.42, 0.51, 0.57, 0.80],
  [3841, 3804, 3788, 3778, 3774],
]

pdp_ck2 = [
  [4,    8,    16,   32,   64,   128],  # note that world size s 64 and pipe spans two GPUs 
  [0.11, 0.11, 0.19, 0.27, 0.27, 0.28],
  [2131, 2131, 2131, 2131, 2131, 2131],
  [0.12, 0.13, 0.22, 0.28, 0.30, 0.33],
  [4450, 4450, 4451, 4451, 4451, 4451],
]

pdp_ck4 = [
  [4,    8,    16,   32,   64,   128],
  [0.17, 0.21, 0.25, 0.32, 0.35, 0.28],
  [1566, 1566, 1566, 1566, 1566, 1566],
  [0.17, 0.19, 0.26, 0.30, 0.35, 0.35],
  [3311, 3311, 3310, 3310, 3310, 3310],
]


def plot_scaling(prefix, bs, blk_size=256, show=True):

  def qps(ws, delay):
    return np.array(ws) * bs * blk_size / np.array(delay) / 1000

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], qps(ddp[0], ddp[1]), '.-', color=COLORS[0])[0],
    plt.plot(ddp[0], qps(ddp[0], ddp[3]), '.--', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], qps(fsdp[0], fsdp[1]), '.-', color=COLORS[1])[0],
    plt.plot(fsdp[0], qps(fsdp[0], fsdp[3]), '.--', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], qps(pdp_ck2[0], pdp_ck2[1]), '.-', color=COLORS[2])[0],
    plt.plot(pdp_ck2[0], qps(pdp_ck2[0], pdp_ck2[3]), '.--', color=COLORS[2])[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper left",
      labels=[
        "ddp bs8", 
        "ddp bs20", 
        "fsdp bs8",
        "fsdp bs20",
        "pdp bs8",
        "pdp bs20",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Number of GPUs")
  plt.ylabel("GPTSmall QPS (1k / Second)")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{prefix}_scaling_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


def plot_memory(prefix, bs, blk_size=256, show=True):

  def gb(mem):
    return np.array(mem) / 1000

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], gb(ddp[2]), '.-', color=COLORS[0])[0],
    plt.plot(ddp[0], gb(ddp[4]), '.--', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], gb(fsdp[2]), '.-', color=COLORS[1])[0],
    plt.plot(fsdp[0], gb(fsdp[4]), '.--', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], gb(pdp_ck2[2]), '.-', color=COLORS[2])[0],
    plt.plot(pdp_ck2[0], gb(pdp_ck2[4]), '.--', color=COLORS[2])[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper right",
      labels=[
        "ddp bs8", 
        "ddp bs20", 
        "fsdp bs8",
        "fsdp bs20",
        "pdp bs8",
        "pdp bs20",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Number of GPUs")
  plt.ylabel("Peak GPU Memory (GB)")
  plt.ylim([0, 40])

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{prefix}_memory_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


plot_scaling("gptsmall", 8, show=False)
plot_memory("gptsmall", 8, show=False)




# GPTLarge
# bs = 8

ddp = [
  [8,     16,    32,    64,    128],  # number of GPUs
  [0.13,  1.15,  1.69,  1.75,  1.67],  # delay
  [7583,  7583,  7583,  7583,  7583],  # memory
  [0.23,  1.12,  1.75,  1.98,  1.70],
  [13722, 13722, 13722, 13722, 13722]
]

fsdp = [
  [8,     16,   32,    64,   128], 
  [0.28,  1.95, 1.96,  2.32, 2.72],
  [4768,  4611, 4470,  4419, 4397],
  [0.30,  2.91, 2.12,  2.40,  2.66],
  [10915, 10766, 10619, 10568, 10543],
]

pdp_ck2 = [
  [4,    8,    16,   32,   64,   128],  # note that world size s 64 and pipe spans two GPUs 
  [0.21, 0.24, 0.80, 1.34, 1.15, 1.36],
  [5517, 5534, 5534, 5534, 5534, 5534],
  [0.35, 0.37, 0.95, 1.35, 1.30, 1.36],
  [9840, 9840, 9849, 9849, 9849, 9849],
]

pdp_ck4 = [
  [4,    8,    16,   32,   64,  128],
  [0.35, 0.37, 0.98, 1.46, 1.40, 1.48],
  [4327, 4327, 4327, 4327, 4327, 4327],
  [0.33, 0.36, 1.07, 1.44, 1.58, 1.65],
  [6754, 6754, 6775, 6775, 6675, 6775],
]


plot_scaling("gptlarge", 8, show=False)
plot_memory("gptlarge", 8, show=False)



# GPTXXL
# bs = 8

ddp = [
  [8,     16,    32,    64,    128],  # number of GPUs
  [0.29,  3.65,  5.20,  5.17,  5.17],  # delay
  [20259, 20259, 20259, 20259, 20259],  # memory
  [0.61,  3.74,  4.69,  5.33,  5.79],
  [33792, 33742, 33742, 33738, 33742],
]

fsdp = [
  [8,     16,    32,    64,    128], 
  [0.37,  7.12,  6.50,  8.20,  8.49],
  [10775, 10091, 9775,  9591,  9505],
  [0.69,  5.91,  6.62,  9.38,  8.89],
  [24192, 23513, 23196, 23006, 22920],
]

pdp_ck2 = [
  [4,     8,     16,    32,    64,    128],  # note that world size s 64 and pipe spans two GPUs 
  [0.42,  0.44,  2.64,  3.94,  4.90,  4.40],
  [13858, 13858, 13858, 13877, 13877, 13858],
  [0.95,  0.98,  3.07,  4.26,  4.63,  4.39],
  [21886, 21890, 21892, 21892, 21892, 21892],
]

pdp_ck4 = [
  [4,     8,     16,    32,    64,    128],
  [0.46,  0.45,  2.45,  4.20,  4.32,  4.72],
  [10050, 10050, 10050, 10050, 11473, 10050],
  [0.87,  0.90,  3.00,  4.54,  4.12,  4.88],
  [15705, 15707, 15726, 15726, 15726, 15726]
]


plot_scaling("gptxxl", 8, show=False)
plot_memory("gptxxl", 8, show=False)























