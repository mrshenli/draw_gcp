import matplotlib.pyplot as plt
import numpy as np



COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
]

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "pdf"

# GPTSmall
# bs = 8

ddp = [
  [8,    16,   32,   64,   128],  # number of GPUs
  [0.08, 0.23, 0.31, 0.30, 0.32],  # delay
  [2195, 2195, 2195, 2195, 2195],  # memory
]

fsdp = [
  [8,    128], 
  [0.15, 0.75],
  [1676, 1613],
]

pdp_ck2 = [
  [128],  # note that world size s 64 and pipe spans two GPUs 
  [0.28],
  [2131],
]

pdp_ck4 = [
  [128],
  [0.28],
  [1566],
]


def plot_scaling(prefix, bs, blk_size=256, show=True):

  def qps(ws, delay):
    return np.array(ws) * bs * blk_size / np.array(delay)

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], qps(ddp[0], ddp[1]), '.-', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], qps(fsdp[0], fsdp[1]), '.-', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], qps(pdp_ck2[0], pdp_ck2[1]), '.-', color=COLORS[2])[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper right",
      labels=[
        "ddp", 
        "fsdp",
        "pdp",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Number of GPUs")
  plt.ylabel("GPTSmall QPS")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{prefix}_scaling_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


def plot_memory(prefix, bs, blk_size=256, show=True):

  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # ddp
    plt.plot(ddp[0], ddp[2], '.-', color=COLORS[0])[0],
    # fsdp
    plt.plot(fsdp[0], fsdp[2], '.-', color=COLORS[1])[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], pdp_ck2[2], '.-', color=COLORS[2])[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper right",
      labels=[
        "ddp", 
        "fsdp",
        "pdp",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Number of GPUs")
  plt.ylabel("Peak GPU Memory")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/{prefix}_memory_bs{bs}.{FIG_TYPE}', bbox_inches='tight')


plot_scaling("gptsmall", 8)
plot_memory("gptsmall", 8)
