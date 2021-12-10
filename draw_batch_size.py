import matplotlib.pyplot as plt
import numpy as np


FONT = {'fontname':'Times New Roman', 'size':12}
FIG_TYPE = "pdf"

"""
GPT 6.7B fp16
16 GPUs
Block size = 256, vocab size = 50000
w/ profiler
"""

prof_fsdp = [
  [2,     4,     6,     8,     12,    14],
  [14.61, 15.48, 13.41, 15.19, 16.34, 16.30],
  [19.24, 22.23, 25.29, 28.31, 34.34, 37.33]
]


prof_fsdp_checkpoint = [
  [8,     16,    24,    32,    40,    48,    56,    64],
  [22.29, 25.85, 28.38, 35.62, 31.39, 26.34, 28.81, 28.42],
  [3.705, 4.523, 5.614, 6.826, 8.036, 9.248, 10.46, 11.67],
]


prof_fsdp_offload = [
  [8,     16,    24,    32,    40,    48,    56,    64],
  [26.11, 28.08, 27.79, 26.29, 29.20, 25.29, 28.12, 28.26],
  [3.168, 3.449, 4.054, 4.746, 5.437, 6.128, 6.820, 7.511],
]

"""
GPT 6.7B fp16
16 GPUs
Block size = 256, vocab size = 50000
w/0 profiler
"""

fsdp = [
  [2,     4,     6,     8,     10,    12,   14],
  [11.57, 11.87, 10.98, 15.19, 13.48, 13.3, 13.30],
  [19.24, 22.23, 25.29, 28.31, 31.30, 34.34, 37.33]
]

fsdp_checkpoint = [
  [8,     16,    24,    32,    40,    48,    56,    64],
  [20.36, 22.66, 22.48, 21.65, 23.65, 23.06, 22.49, 27.11],
  [3.705, 4.523, 5.614, 6.826, 8.036, 9.248, 10.46, 11.67],
]

fsdp_offload = [
  [8,     16,    24,    32,    40,    48,    56,    64],
  [23.07, 27.79, 21.85, 22.15, 24.99, 22.25, 22.06, 22.57],
  [3.168, 3.449, 4.054, 4.746, 5.437, 6.128, 6.820, 7.511],
]


def plot_profiler(show=False):
  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # w/ profiler
    plt.plot(prof_fsdp[0], prof_fsdp[1], '.-', color='black')[0],
    plt.plot(prof_fsdp_checkpoint[0], prof_fsdp_checkpoint[1], 'x-', color='black')[0],
    plt.plot(prof_fsdp_offload[0], prof_fsdp_offload[1], '^-', color='black')[0],
    # w/o profiler
    plt.plot(fsdp[0], fsdp[1], '.-', color='red')[0],
    plt.plot(fsdp_checkpoint[0], fsdp_checkpoint[1], 'x-', color='red')[0],
    plt.plot(fsdp_offload[0], fsdp_offload[1], '^-', color='red')[0],
  ])

  plt.legend(
      handles=handles,
      loc="lower right",
      labels=[
        "pf fsdp", 
        "pf fsdp cp",
        "pf fsdp ol",
        "fsdp", 
        "fsdp cp",
        "fsdp ol",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel("Per-GPU Batch Size (GPT 6.7B, 16 GPUs)")
  plt.ylabel("Delay (S)")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/fsdp_profiler.{FIG_TYPE}', bbox_inches='tight')


plot_profiler()


pdp_ck2 = [
  [2,    4,    6,    8,    10,   12,   14,   16,   18,   20], # per-GPU batch size
  [5.84, 6.26, 6.15, 6.43, 6.86, 6.36, 6.81, 7.09, 6.99, 7.03], # delay
  [22.7, 24.5, 26.3, 28.0, 29.7, 31.5, 33.2, 35.0, 36.7, 38.5], # memory
]


pdp_ck4 = [
  [2,    4,    6,    8,    10,   12,   14,   16,   18,   20,  22],
  [6.47, 6.35, 6.22, 6.63, 6.29, 7.62, 6.66, 7.07, 7.79, 7.52, 7.49],
  [21.9, 22.8, 23.8, 24.7, 25.6, 26.6, 27.5, 28.4, 29.3, 30.2, 31.2]
]



def plot_delay(ws, show=False):
  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # fsdp w/o profiler
    plt.plot(fsdp[0], fsdp[1], '.-', color='red')[0],
    plt.plot(fsdp_checkpoint[0], fsdp_checkpoint[1], 'x-', color='red')[0],
    plt.plot(fsdp_offload[0], fsdp_offload[1], '^-', color='red')[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], pdp_ck2[1], '.-', color='blue')[0],
    plt.plot(pdp_ck2_checkpoint[0], pdp_ck2_checkpoint[1], 'x-', color='blue')[0],
    plt.plot(pdp_ck2_offload[0], pdp_ck2_offload[1], '^-', color='blue')[0],
    # pdp ck=4
    plt.plot(pdp_ck4[0], pdp_ck4[1], '.--', color='grey')[0],
    plt.plot(pdp_ck4_checkpoint[0], pdp_ck4_checkpoint[1], 'x--', color='grey')[0],
    plt.plot(pdp_ck4_offload[0], pdp_ck4_offload[1], '^--', color='grey')[0],
  ])

  plt.legend(
      handles=handles,
      loc="lower right",
      labels=[
        "fsdp", 
        "fsdp cp",
        "fsdp ol",
        "pdp2",
        "pdp2 cp",
        "pdp2 ol",
        "pdp4",
        "pdp4 cp",
        "pdp4 ol",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Per-GPU Batch Size (GPT 6.7B, {ws} GPUs)")
  plt.ylabel("Delay (S)")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/fsdp_pdp_delay_ws{ws}.{FIG_TYPE}', bbox_inches='tight')


#plot_delay(16)


def plot_mem(ws, show=False):
  plt.figure(figsize=(6, 3))
  handles = []
  handles.extend([
    # fsdp w/o profiler
    plt.plot(fsdp[0], fsdp[2], '.-', color='red')[0],
    plt.plot(fsdp_checkpoint[0], fsdp_checkpoint[2], 'x-', color='red')[0],
    plt.plot(fsdp_offload[0], fsdp_offload[2], '^-', color='red')[0],
    # pdp ck=2
    plt.plot(pdp_ck2[0], pdp_ck2[2], '.-', color='blue')[0],
    plt.plot(pdp_ck2_checkpoint[0], pdp_ck2_checkpoint[2], 'x-', color='blue')[0],
    plt.plot(pdp_ck2_offload[0], pdp_ck2_offload[2], '^-', color='blue')[0],
    # pdp ck=4
    plt.plot(pdp_ck4[0], pdp_ck4[2], '.--', color='grey')[0],
    plt.plot(pdp_ck4_checkpoint[0], pdp_ck4_checkpoint[2], 'x--', color='grey')[0],
    plt.plot(pdp_ck4_offload[0], pdp_ck4_offload[2], '^--', color='grey')[0],
  ])

  plt.legend(
      handles=handles,
      loc="upper right",
      labels=[
        "fsdp", 
        "fsdp cp",
        "fsdp ol",
        "pdp2",
        "pdp2 cp",
        "pdp2 ol",
        "pdp4",
        "pdp4 cp",
        "pdp4 ol",
      ],
      prop={'family':FONT['fontname'], 'size':FONT['size']},
      ncol=2,
      #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
  )


  plt.xlabel(f"Per-GPU Batch Size (GPT 6.7B, {ws} GPUs)")
  plt.ylabel("Memory (GB)")

  if show:
    plt.show()
  else:
    plt.savefig(f'image/fsdp_pdp_mem_ws{ws}.{FIG_TYPE}', bbox_inches='tight')


#plot_mem(16)

"""
GPT 6.7B fp16
128 GPUs
Block size = 256, vocab size = 50000
w/0 profiler
"""

fsdp = [
  [2,     4,     6,     8,     10,    12,    14,    16],
  [18.00, 19.07, 19.30, 19.53, 19.37, 23.39, 23.57, 26.68],
  [4.533, 7.542, 10.56, 13.57, 16.59, 19.60, 22.61, 25.66]
]

fsdp_offload = [
  [8,     16,    32,    40,    48,    56],
  [26.07, 24.44, 25.06, 29.01, 26.80, 25.79],
  [1.733, 2.014, 3.291, 3.982, 4.674, 5.366],
]

fsdp_checkpoint = [
  [8,     16,    24,    32,    40,    48,    56,    64],
  [24.54, 24.78, 27.83, 25.65, 27.24, 25.73, 27.06, 28.87],
  [2.270, 3.088, 4.160, 5.372, 6.582, 7.794, 9.006, 10.22],
]

pdp_ck2 = [
  [2,     4,     6,     8,     10,    12,    14,    16,    18,    20],
  [11.00, 10.99, 10.59, 11.31, 11.28, 10.75, 11.08, 10.73, 12.62, 13.16],
  [22.72, 24.47, 26.22, 27.97, 29.72, 31.47, 33.22, 34.97, 36.72, 38.47]
]

pdp_ck2_checkpoint = [
  [4,     8,     12,    16,    20,    24],
  [10.57, 12.08, 10.78, 11.09, 12.27, 14.97],
  [21.76, 22.54, 23.32, 24.10, 24.88, 25.67],
]

pdp_ck2_offload = [
  [4,     8,     12,    16,    20,    24],
  [11.90, 10.82, 12.42, 12.60, 11.99, 13.02],
  [21.62, 22.27, 22.92, 23.57, 24.21, 24.86],
]


pdp_ck4 = [
  [2,     4,    6,     8,     10,    12,    14,    16,    18,    20,   22],
  [11.70, 9.600, 11.94, 11.06, 10.93, 11.53, 12.68, 12.29, 11.25, 11.82, 10.71],
  [21.91, 22.83, 23.79, 24.70, 25.64, 26.55, 27.49, 28.42, 29.35, 30.27, 31.21],
]

pdp_ck4_checkpoint = [
  [4,     8,     12,    16,    20,    24],
  [10.97, 11.19, 11.18, 12.55, 12.14, 12.54],
  [21.47, 21.98, 22.48, 22.98, 23.48, 23.98],
]

pdp_ck4_offload = [
  [4,     8,     12,    16,    20,    24],
  [11.30, 12.28, 11.65, 11.95, 12.10, 12.40],
  [21.41, 21.84, 22.28, 22.71, 23.15, 23.58],
]


plot_delay(128)
plot_mem(128)
































