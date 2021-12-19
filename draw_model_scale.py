import matplotlib.pyplot as plt
import numpy as np



COLORS = [
  "#283350",
  "#f93800",
  "#ffb500",
  "#888888",
]

FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "png"


# world size = 32, per-GPU batch size = 16

ddp = [
  [125,  350,  760,   1300,  2700],  # GPTSmall, Medium, Large, Xl, XXL
  [0.28, 0.75, 1.43,  2.55,  5.13],  # per-iteration delay
  [3625, 8014, 11548, 16748, 29088], # Peak mem
]

ddp_checkpoint = [
  [125,  350,  760,  1300, 2700],
  [0.31, 0.85, 1.43, 2.93, 4.22],
  [2069, 3186, 5660, 9278, 17425],
]

ddp_offload = [
  [125,  350,  760,  1300, 2700],
  [0.29, 0.71, 1.31, 2.97, 4.95],
  [1994, 3047, 5649, 9278, 17427],
]

fsdp = [
  [125,  350,  760,  1300,  2700,  6700],  # GPTSmall, Large, Xl, XXL, XXXL
  [0.46, 1.24, 2.77, 3.55,  7.35,  22.89],
  [3056, 6543, 8423, 11416, 18555, 26230],
]

fsdp_checkpoint = [
  [125,  350,  760,  1300,  2700,  6700,  13000, 39000,  76000],
  [0.55, 1.45, 2.75, 5.01,  8.66,  25.85, 52.43, 143.14, 319.67],
  [1091, 1306, 1575, 1891,  2434,  3678,  6136,  10649,  20770],
]

fsdp_offload = [
  [125,  350,  760,  1300,  2700,  6700,  13000, 39000,  76000],
  [0.54, 1.26, 2.62, 7.20,  8.63,  20.79, 60.18, 141.54, 322.60],
  [1016, 1104, 1273, 1485,  1763,  2604,  4152,  7692,   15429],
]


pdp_ck2 = [  # for pdp, it's world size = 16 and batch size = 32
  [125,  350,  760,  1300,  2700,  6700],
  [0.30, 0.77, 1.40, 2.37,  4.48,  10.46],
  [3675, 6120, 8353, 11590, 19134, 34969],
]

pdp_ck2_checkpoint = [
  [125,  350,  760,  1300, 2700,  6700],
  [0.31, 0.71, 1.56, 2.92, 4.48,  10.36],
  [3095, 3912, 5341, 7240, 11490, 24102],
]

pdp_ck2_offload = [
  [125,  350,  760,  1300, 2700,  6700],
  [0.32, 0.81, 1.52, 2.44, 4.47,  11.21],
  [3059, 3811, 5191, 7037, 11153, 23565],
]

pdp_ck4 = [
  [125,  350,  760,  1300, 2700,  6700],  
  [0.33, 0.87, 1.41, 2.35, 4.19,  11.78],
  [2707, 4140, 5959, 8478, 14310, 28416],
]

pdp_ck4_checkpoint = [
  [125,  350,  760,  1300, 2700,  6700], 
  [0.39, 0.95, 1.44, 2.46, 4.40,  10.70],
  [2257, 3036, 4434, 6304, 10484, 22982],
]

pdp_ck4_offload = [
  [125,  350,  760,  1300, 2700,  6700], 
  [0.40, 0.95, 1.67, 2.56, 4.65,  12.07],
  [2236, 2984, 4360, 6202, 10314, 22714],
]



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


plot_qps(32, 16, show=False)
plot_mem(16, show=False)



























