import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np



COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
]

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"

INF = float("inf")

# GPTSmall

ddp = [
  [1,    2,    4,    8,    16,   32,   64,   128],  # number of GPUs
  [0.06, 0.06, 0.06, 0.07, 0.23, 0.31, 0.30, 0.32],  # delay bs = 8
  [2192, 2192, 2191, 2195, 2195, 2195, 2195, 2195],  # memory
  [0.07, 0.07, 0.08, 0.09, 0.23, 0.31, 0.31, 0.36],  # delay bs = 20
  [4350, 4352, 4350, 4355, 4355, 4355, 4355, 4355],  # memory
]

fsdp = [
  [1,    2,    4,    8,    16,   32,   64,   128], 
  [0.09, 0.14, 0.14, 0.15, 0.40, 0.47, 0.58, 0.75],
  [2185, 1895, 1751, 1676, 1642, 1625, 1616, 1613],  
  [0.09, 0.14, 0.15, 0.18, 0.42, 0.51, 0.57, 0.80],
  [4356, 4057, 3916, 3841, 3804, 3788, 3778, 3774],
]

pdp_ck2 = [
  [1,    2,    4,    8,    16,   32,   64,   128],  # note that world size s 64 and pipe spans two GPUs 
  [INF,  0.10, 0.11, 0.11, 0.19, 0.27, 0.27, 0.28], # pipeline needs at least 2 devices
  [INF,  2131, 2131, 2131, 2131, 2131, 2131, 2131], # hence INF for 1 GPU column
  [INF,  0.11, 0.12, 0.13, 0.22, 0.28, 0.30, 0.33],
  [INF,  4450, 4450, 4450, 4451, 4451, 4451, 4451],
]

pdp_ck4 = [
  [4,    8,    16,   32,   64,   128],
  [0.17, 0.21, 0.25, 0.32, 0.35, 0.28],
  [1566, 1566, 1566, 1566, 1566, 1566],
  [0.17, 0.19, 0.26, 0.30, 0.35, 0.35],
  [3311, 3311, 3310, 3310, 3310, 3310],
]

opt = [
  [1,    2,    4,    8,    16,   32,   64,   128],
  [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
  [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
]


def plot_qps(model, bs, blk_size=256, show=True):

  def qps(ws, bs, delay):
    return np.array(ws) * bs * blk_size / np.array(delay) / 1000

  plt.figure(figsize=(8, 3))

  width = 0.12
  x = np.arange(8)
  handles = [
    plt.bar(x - 2.5 * width, qps(ddp[0], bs[0], ddp[1]), width, color=COLORS[0], label='ddp8')[0], 
    plt.bar(x - 1.5 * width, qps(ddp[0], bs[1], ddp[3]), width, color=COLORS[0], label='ddp20', hatch="///", edgecolor="w")[0], 
    plt.bar(x - 0.5 * width, qps(pdp_ck2[0], bs[0], pdp_ck2[1]), width, color=COLORS[2], label='pdp8')[0], 
    plt.bar(x + 0.5 * width, qps(pdp_ck2[0], bs[1], pdp_ck2[3]), width, color=COLORS[2], label='pdp20', hatch="///", edgecolor="w")[0],
    plt.bar(x + 1.5 * width, qps(fsdp[0], bs[0], fsdp[1]), width, color=COLORS[1], label='fsdp8')[0], 
    plt.bar(x + 2.5 * width, qps(fsdp[0], bs[1], fsdp[3]), width, color=COLORS[1], label='fsdp20', hatch="///", edgecolor="w")[0],
  ]

  opt8_qps  = qps(opt[0], bs[0], opt[1])
  opt20_qps = qps(opt[0], bs[1], opt[2])
  for i, y in enumerate(opt8_qps):
    plt.hlines(y=y, xmin=i-3*width, xmax=i+3*width, color="grey", linestyle="--")

  opt8_line = mlines.Line2D([], [], color='grey', linestyle='--', label='opt8')
  handles.append(opt8_line)

  for i, y in enumerate(opt20_qps):
    plt.hlines(y=y, xmin=i-3*width, xmax=i+3*width, color="grey", linestyle="-")

  opt20_line = mlines.Line2D([], [], color='grey', linestyle='-', label='opt20')
  handles.append(opt20_line)

  plt.legend(
      handles=handles,
      labels=[
        "ddp8",
        "ddp20",
        "pdp8",
        "pdp20",
        "fsdp8",
        "fsdp20",
        "opt8",
        "opt20",
      ],
      loc="upper left",
      #prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=4,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )

  ax = plt.gca()
  ax.set_xticks(x)
  ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])

  plt.xlabel(f"Number of GPUs")
  plt.ylabel(f"{model} QPS (1k / Second)")
  #plt.grid(True, which="both")
  plt.yscale("log")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{model}_scaling_bs{bs}.{FIG_TYPE}', bbox_inches='tight')



def plot_scaling(model, bs, blk_size=256, show=True):

  def qps(ws, bs, delay):
    return np.array(ws) * bs * blk_size / np.array(delay) / 1000

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], qps(ddp[0], bs[0], ddp[1]), '.-', color=COLORS[0])[0],
    plt.plot(ddp[0], qps(ddp[0], bs[1], ddp[3]), '.--', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], qps(fsdp[0], bs[0], fsdp[1]), '.-', color=COLORS[1])[0],
    plt.plot(fsdp[0], qps(fsdp[0], bs[1], fsdp[3]), '.--', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], qps(pdp_ck2[0], bs[0], pdp_ck2[1]), '.-', color=COLORS[2])[0],
    plt.plot(pdp_ck2[0], qps(pdp_ck2[0], bs[1], pdp_ck2[3]), '.--', color=COLORS[2])[0],
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
  plt.ylabel(f"{model} QPS (1k / Second)")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{model}_scaling_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


def plot_memory(model, bs, blk_size=256, show=True):

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
  plt.ylabel(f"{model} Peak GPU Memory (GB)")
  plt.ylim([0, 40])

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{model}_memory_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


#plot_scaling("GPTSmall", [8, 20], show=False)
#plot_memory("GPTSmall", [8, 20], show=False)

plot_qps("GPTSmall", [8, 20], show=False)


# GPTLarge
# bs = 8

ddp = [
  [1,     2,     4,     8,     16,    32,    64,    128],  # number of GPUs
  [0.11,  0.11,  0.12,  0.12,  1.15,  1.69,  1.75,  1.67],  # delay
  [7566,  7567,  7557,  7583,  7583,  7583,  7583,  7583],  # memory
  [0.21,  0.22,  0.23,  0.23,  1.12,  1.75,  1.98,  1.70],
  [13728, 13728, 13719, 13750, 13722, 13722, 13722, 13722]
]

fsdp = [
  [1,     2,     4,     8,     16,   32,    64,   128], 
  [0.16,  0.26,  0.27,  0.28,  1.95, 1.96,  2.32, 2.72],
  [7554,  5962,  5166,  4768,  4611, 4470,  4419, 4397],
  [0.23,  0.27,  0.28,  0.30,  2.91, 2.12,  2.40,  2.66],
  [13755, 12112, 11315, 10915, 10766, 10619, 10568, 10543],
]

pdp_ck2 = [
  [1,    2,    4,    8,    16,   32,   64,   128],  # note that world size s 64 and pipe spans two GPUs 
  [INF,  0.18, 0.21, 0.24, 0.80, 1.34, 1.15, 1.36],
  [INF,  5519, 5517, 5534, 5534, 5534, 5534, 5534],
  [INF,  0.34, 0.35, 0.37, 0.95, 1.35, 1.30, 1.36],
  [INF,  9840, 9840, 9840, 9849, 9849, 9849, 9849],
]

pdp_ck4 = [
  [4,    8,    16,   32,   64,  128],
  [0.35, 0.37, 0.98, 1.46, 1.40, 1.48],
  [4327, 4327, 4327, 4327, 4327, 4327],
  [0.33, 0.36, 1.07, 1.44, 1.58, 1.65],
  [6754, 6754, 6775, 6775, 6675, 6775],
]


opt = [
  [1,    2,    4,    8,    16,   32,   64,   128],
  [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
  [0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21],
]


#plot_scaling("GPTLarge", [8, 20], show=False)
#plot_memory("GPTLarge", [8, 20], show=False)

plot_qps("GPTLarge", [8, 20], show=False)

# GPTXXL
# bs = 8

ddp = [
  [1,     2,     4,     8,     16,    32,    64,    128],  # number of GPUs
  [0.25,  0.28,  0.28,  0.29,  3.65,  5.20,  5.17,  5.17],  # delay
  [20233, 20228, 20236, 20259, 20259, 20259, 20259, 20259],  # memory
  [0.57,  0.60,  0.60,  0.61,  3.74,  4.69,  5.33,  5.79],
  [33779, 33783, 33787, 33792, 33742, 33742, 33738, 33742],
]

fsdp = [
  [1,     2,     4,     8,     16,    32,    64,    128], 
  [0.26,  0.35,  0.35,  0.37,  7.12,  6.50,  8.20,  8.49],
  [20242, 14946, 12120, 10775, 10091, 9775,  9591,  9505],
  [0.59,  0.66,  0.67,  0.69,  5.91,  6.62,  9.38,  8.89],
  [33746, 28365, 25546, 24192, 23513, 23196, 23006, 22920],
]

pdp_ck2 = [
  [1,     2,     4,     8,     16,    32,    64,    128],  # note that world size s 64 and pipe spans two GPUs 
  [INF,   0.39,  0.42,  0.44,  2.64,  3.94,  4.90,  4.40],
  [INF,   13854, 13858, 13858, 13858, 13877, 13877, 13858],
  [INF,   0.92,  0.95,  0.98,  3.07,  4.26,  4.63,  4.39],
  [INF,   21892, 21886, 21890, 21892, 21892, 21892, 21892],
]

pdp_ck4 = [
  [4,     8,     16,    32,    64,    128],
  [0.46,  0.45,  2.45,  4.20,  4.32,  4.72],
  [10050, 10050, 10050, 10050, 11473, 10050],
  [0.87,  0.90,  3.00,  4.54,  4.12,  4.88],
  [15705, 15707, 15726, 15726, 15726, 15726]
]


opt = [
  [1,    2,    4,    8,    16,   32,   64,   128],
  [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
  [0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57],
]


#plot_scaling("GPT2.7B", [8, 20], show=False)
#plot_memory("GPT2.7B", [8, 20], show=False)

plot_qps("GPT2.7B", [8, 20], show=False)























