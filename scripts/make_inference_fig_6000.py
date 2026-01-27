# /// script
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib


markersize = 3
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['xtick.major.size'] = 3.5
matplotlib.rcParams['ytick.major.size'] = 3.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = markersize
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.framealpha'] = 1.0
matplotlib.rcParams['legend.edgecolor'] = 'black'
matplotlib.rcParams['legend.fancybox'] = False
matplotlib.rcParams['legend.fontsize'] = 8

cmap_autumn = plt.get_cmap('autumn')
cmap_winter = plt.get_cmap('winter')

# our_model = "Nequix-MP-1.5"
atom_name = "Si"
lattice_constant = 5.43
compliant_status = ""

def _series(df: pd.DataFrame, model_name: str) -> tuple[list[int], list[float]]:
    sub = df[df["model"] == model_name].copy()
    sub = sub[sub["num_atoms"] != 8]
    sub = sub.sort_values("num_atoms")
    return sub["num_atoms"].tolist(), sub["steps_per_day_m"].tolist()


plot_kwargs = {'marker': 's', 'markersize': markersize, 'linestyle': '-', 'markeredgecolor':'black'}
def make_inference_fig(path: str) -> None:
    
    atom_name = path.split("_")[3]
    lattice_constant = path.split("_")[4][:-4]

    df = pd.read_csv(path)
    gpu_name = df["gpu"].unique()[0]
    precision = df["precision"].unique()[0]
    df["steps_per_day_m"] = 86_400_000.0 / df["time"] / 1_000_000.0

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    # Orb-v3-cons-inf-omat-highest 
    x, y_millions = _series(df, "Orb-v3-cons-inf-omat")
    ax.plot(x, y_millions, color=cmap_winter(0.6), label="Orbv3-highest", **plot_kwargs)

    # Orb-v3-cons-inf-omat-highest + cuml 
    x, y_millions = _series(df, "Orb-v3-cons-inf-omat-cuml")
    ax.plot(x, y_millions, color=cmap_winter(0.5), label="Orbv3-highest-cuml", **plot_kwargs)

    # Orb-v3-cons-inf-omat-high 
    x, y_millions = _series(df, "Orb-v3-cons-inf-omat-high")
    ax.plot(x, y_millions, color=cmap_winter(0.4), label="Orbv3-high", **plot_kwargs)
 
    # Orb-v3-cons-inf-omat-high + cuml 
    x, y_millions = _series(df, "Orb-v3-cons-inf-omat-high-cuml")
    ax.plot(x, y_millions, color=cmap_winter(0.3), label="Orbv3-high-cuml", **plot_kwargs)

    # MACE-OMAT-0-medium
    x, y_millions = _series(df, "MACE-OMAT-0")  # somewhat slow, might be optimizable
    ax.plot(x, y_millions, color="fuchsia", label="MACE-OMAT-0", **plot_kwargs)
    mace_x = x.copy()

    # NequIP-OAM-L
    x, y_millions = _series(df, "NequIP-MP-L")  # label typo
    ax.plot(x, y_millions, color="green", label="NequIP-OAM-L", **plot_kwargs)

    # SevenNet-l3i5 (no acceleration)
    x, y_millions = _series(df, "SevenNet-l3i5")
    ax.plot(x, y_millions, color=cmap_autumn(0.6), label="7net-l3i5_van", **plot_kwargs)

    # SevenNet-l3i5-cueq
    x, y_millions = _series(df, "SevenNet-l3i5-cueq")
    ax.plot(x, y_millions, color=cmap_autumn(0.5), label="7net-l3i5", **plot_kwargs)

    # SevenNet-Omni-cueq
    x, y_millions = _series(df, "SevenNet-Omni-cueq")
    ax.plot(x, y_millions, color=cmap_autumn(0.4), label="7net-omni", **plot_kwargs)

    # SevenNet-Omni-i8-cueq
    x, y_millions = _series(df, "SevenNet-Omni-i8-cueq")
    ax.plot(x, y_millions, color=cmap_autumn(0.3), label="7net-omni-i8", **plot_kwargs)

    # SevenNet-Omni-i12-cueq
    x, y_millions = _series(df, "SevenNet-Omni-i12-cueq")
    ax.plot(x, y_millions, color=cmap_autumn(0.2), label="7net-omni-i12", **plot_kwargs)

 

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    x_labels = sorted(set(mace_x))
    ax.xaxis.set_major_locator(FixedLocator(x_labels))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in x_labels]))
    ax.tick_params(axis="x", labelrotation=75, labelsize=6)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    # Style grid like the paper
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=5, loc='best')
    legend.get_frame().set_linewidth(0.5)
    
    # ax.set_title(f"Matbench {compliant_status}, {atom_name} diamond lattice_constant={lattice_constant} Å \n {gpu_name} {precision}", fontsize=8, weight='normal')

    fig.tight_layout()
    fig.savefig(f"./figures/inference_fig_{compliant_status}_{atom_name}_{lattice_constant}_{gpu_name}_{precision}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"./figures/inference_fig_{compliant_status}_{atom_name}_{lattice_constant}_{gpu_name}_{precision}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
 
    make_inference_fig(f"./data/timing_data_pro6000_{atom_name}_{lattice_constant}.csv")

if __name__ == "__main__":
    main()
