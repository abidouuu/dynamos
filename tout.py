# Fichier général qui fait tous les plots de l'article. Une fonction par graphique. Harmonisation de l'esthétisme.

import numpy as np
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm
from math import sqrt
from scipy.signal import find_peaks
from matplotlib.cm import Blues, Oranges, Greens
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency")
warnings.filterwarnings("ignore", message='Creating legend with loc="best" can be slow with large amounts of data.')
import logging
logging.getLogger('matplotlib.backends.backend_ps').setLevel(logging.ERROR)
from plot_style import PLOT_STYLE

workdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(workdir, "data")


# ==========================================================================================================
# -------------- FONCTIONS UTILITAIRES (facteur commun à toutes les figures) -------------------------------
# ==========================================================================================================

def new_figure(figsize=None, nrows=1, ncols=1, sharex=False):
    if figsize is None:
        figsize = PLOT_STYLE["figsize"]

    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex
    )

def style_axis(
        ax,
        xlabel=None,
        ylabel=None,
        title=None,
        legend=True,
        grid=True,
        xscale=None,
        yscale=None):

    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=PLOT_STYLE["label_fontsize"]
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=PLOT_STYLE["label_fontsize"]
        )

    if title is not None:
        ax.set_title(
            title,
            fontsize=PLOT_STYLE["title_fontsize"]
        )

    ax.tick_params(
        axis='both',
        labelsize=PLOT_STYLE["tick_fontsize"]
    )

    if grid:
        ax.grid(
            True,
            alpha=PLOT_STYLE["grid_alpha"]
        )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                fontsize=PLOT_STYLE["legend_fontsize"]
            )

    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)


def save_fig(fig, folder, filename,
             dpi=None,
             bbox_inches="tight",
             close=True):

    if dpi is None:
        dpi = PLOT_STYLE["dpi"]

    os.makedirs(folder, exist_ok=True)

    base = os.path.join(folder, filename)

    for ext in PLOT_STYLE["save_formats"]:
        fig.savefig(
            base + "." + ext,
            dpi=dpi,
            bbox_inches=bbox_inches
        )

    if close:
        plt.close(fig)

# ==========================================================================================================
# -------------- FIGURES 1 & 2 - Solutions analytiques -----------------------------------------------------
# ==========================================================================================================
datadir_short = os.path.join(datadir, "1_Court_terme_analytique")
datadir_fig1 = os.path.join(datadir_short, "Figure_1_balaye_kappa")
datadir_fig2 = os.path.join(datadir_short, "Figure_2_balaye_epsilon")


def balayage(vary_epsilon):  # x varie continûment, y quelques courbes.
    Lambdas = [0.001, 0.01, 0.1, 1]
    epsilons_x = np.linspace(-2, 2, 10000)
    kappas_x = 10 ** np.linspace(-3, 2, 10000)
    epsilons_y = [-0.1, -0.01, 0, 0.01, 0.1]
    kappas_y = 10.0 ** np.arange(-2, 2)

    if vary_epsilon:
        xs = epsilons_x
        ys = kappas_y
        datadir_ = datadir_fig2
        print("Balayage de epsilon : start")
    else:
        xs = kappas_x
        ys = epsilons_y
        datadir_ = datadir_fig1
        print("Balayage de kappa : start")

    for Lambda in tqdm(Lambdas, desc="Lambda", position=0):
        cfg = config(datadir=datadir_, simu_title="Lambda=" + str(Lambda) + "_")
        cfg.Lambda = Lambda

        fig, (axB, axb) = new_figure(figsize=PLOT_STYLE["figsize_double_tall"], nrows=2, sharex=True)
        xscale = None if vary_epsilon else 'log'

        blues = Blues(np.linspace(0.35, 0.95, len(ys)))
        oranges = Oranges(np.linspace(0.35, 0.95, len(ys)))

        if not vary_epsilon:
            greens_bool = [a >= 0 for a in ys]
            greens = Greens(np.linspace(0.35, 0.95, len(greens_bool)))

        first = True
        # FIX: green_idx must be initialized once, OUTSIDE the y-loop, so that
        # each non-negative-epsilon curve gets a distinct color from the gradient
        # instead of every curve reusing greens[0].
        green_idx = 0

        for y_idx, y in enumerate(ys):
            B, b = [], []
            B_landau, b_landau = [], []

            if vary_epsilon:
                cfg.kappaeq = y
            else:
                cfg.epsiloneq = y

            for x in xs:
                if vary_epsilon:
                    cfg.epsiloneq = x
                else:
                    cfg.kappaeq = x

                (B_eq, b_eq) = cfg.get_eq()[0]
                B.append(B_eq)
                b.append(b_eq)
                if cfg.epsiloneq > 0:
                    B_landau.append(sqrt(cfg.epsiloneq / Lambda))
                else:
                    B_landau.append(0)
                b_landau.append(1)

            if vary_epsilon:
                label = rf"$\kappa={y:.1e}$"
                axB.plot(xs, B_landau, color='green', ls="--", lw=PLOT_STYLE["linewidth"], label="B_landau" if first else None)
                axb.plot(xs, b_landau, color='green', ls="--", lw=PLOT_STYLE["linewidth"], label="b_landau" if first else None)
                first = False
            else:
                label = rf"$\varepsilon={y:.1e}$"
                axb.plot(xs, b_landau, color='green', ls="--", lw=PLOT_STYLE["linewidth"], label="b_landau" if first else None)
                if y >= 0:
                    axB.plot(xs, B_landau, color=greens[green_idx], ls="--", lw=PLOT_STYLE["linewidth"], label="B_landau : " + label)
                    green_idx += 1

            axB.plot(xs, B, color=blues[y_idx], lw=PLOT_STYLE["linewidth"], label=label)
            axb.plot(xs, b, color=oranges[y_idx], lw=PLOT_STYLE["linewidth"], label=label)

        style_axis(axB, ylabel="B", xscale=xscale)
        style_axis(axb, xlabel=r"$\varepsilon$" if vary_epsilon else r"$\kappa$", ylabel="b", xscale=xscale)

        fig.tight_layout()
        cfg.write_config_file()
        save_fig(fig, cfg.folder, f"Lambda={Lambda}")

    if vary_epsilon: print("Balayage de epsilon : fini")
    else: print("Balayage de kappa : fini")


def fig_1(): balayage(False)
def fig_2(): balayage(True)

def court_terme():
    fig_1()
    fig_2()


# ==========================================================================================================
# -------------- FIGURES 3,4,5,6 : Moyen terme, intermittence ---------------------------------------------
# ==========================================================================================================
datadir_mid = os.path.join(datadir, "2_Moyen_terme_Intermittence")
Lambdas = [0.01,0.1]
epsiloneqs = [-0.1, 0, 0.1]
kappaeqs = [0, 0.1, 1]
inter_list = [(True, False), (False, True), (True,True)]  # kappa, epsilon


def plot_fig5(epsilon_data, kappa_data, B, folder):
    n_parts = 10
    n = len(epsilon_data)

    # length of each chunk
    chunk_size = n // n_parts

    for i in range(n_parts):
        start = i * chunk_size

        # make sure the last chunk takes the remaining points
        if i == n_parts - 1:
            end = n
        else:
            end = (i + 1) * chunk_size

        eps = epsilon_data[start:end]
        kap = kappa_data[start:end]
        B_part = B[start:end]

        plot_fig5_part(eps, kap, B_part, folder, i+1)

def plot_fig5_part(epsilon_data, kappa_data, B, folder, part_number):
    """Plot one portion of Figure 5."""

    fig_5, ax_5 = new_figure()
    fig_5.patch.set_facecolor('white')
    ax_5.set_facecolor('white')

    xlim = (epsilon_data.min(), epsilon_data.max())
    ylim = (kappa_data.min(), kappa_data.max())

    ax_5.set_xlim(xlim)
    ax_5.set_ylim(ylim)

    # zone grisée : kappa + epsilon < 0
    x_fill = np.linspace(xlim[0], xlim[1], 500)
    y_boundary = -x_fill
    ax_5.fill_between(
        x_fill,
        ylim[0],
        y_boundary,
        color='0.85',
        zorder=0
    )

    # trajectoire colorée selon B
    points = np.array([epsilon_data, kappa_data]).T.reshape(-1, 1, 2)
    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1
    )

    norm = plt.Normalize(B.min(), B.max())

    lc = LineCollection(
        segments,
        cmap='coolwarm',
        norm=norm,
        zorder=2
    )

    lc.set_array(B[:-1])
    lc.set_linewidth(PLOT_STYLE["linewidth"])

    line = ax_5.add_collection(lc)

    cbar = fig_5.colorbar(line, ax=ax_5)
    cbar.set_label(r"$B$")

    style_axis(
        ax_5,
        xlabel=r"$\varepsilon(t)$",
        ylabel=r"$\kappa(t)$",
        legend=False
    )

    ax_5.grid(True, zorder=1, alpha=0.3)

    fig_5.tight_layout()

    save_fig(
        fig_5,
        os.path.join(folder,"trajectoires"),
        f"Fig_5_trajectoire_part_{part_number}",
        close=True
    )

def plot_fft(t, signal, label, folder, filename_prefix, max_freq=0.1):
    dt = t[1] - t[0]
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(signal - signal.mean())
    amplitude = np.abs(fft_vals)

    freq_mask = freqs <= max_freq
    freqs_filtered = freqs[freq_mask]
    amplitude_filtered = amplitude[freq_mask]

    fig_fft, ax_fft = new_figure()
    ax_fft.plot(freqs_filtered, amplitude_filtered, lw=PLOT_STYLE["linewidth"], color='purple')
    style_axis(ax_fft, xlabel="Fréquence", ylabel="Amplitude", xscale='log', yscale='log',
               title=f"FFT de {label}", legend=False)
    save_fig(fig_fft, folder, f"{filename_prefix}_fft", dpi=300)

    peaks, _ = find_peaks(amplitude_filtered)

    if len(peaks) == 0:
        top_freqs = [freqs_filtered[np.argmax(amplitude_filtered)]]
        top_amps = [amplitude_filtered.max()]
    else:
        peak_amps = amplitude_filtered[peaks]
        peak_freqs = freqs_filtered[peaks]
        n_top = min(3, len(peaks))
        top_idx = np.argsort(peak_amps)[::-1][:n_top]
        top_freqs = peak_freqs[top_idx]
        top_amps = peak_amps[top_idx]

    with open(os.path.join(folder, f"{filename_prefix}_fft_top3.txt"), "w") as f:
        f.write(f"Fréquences dominantes de la FFT de {label}\n")
        f.write("=" * 50 + "\n")
        if len(peaks) == 0:
            f.write("Aucun pic local détecté, fréquence du maximum global reportée :\n")
        for rank, (freq, amp) in enumerate(zip(top_freqs, top_amps), start=1):
            f.write(f"{rank}. Fréquence = {freq:.6e}   |   Amplitude = {amp:.6e}\n")

def plot_fig4(t, B, b, kappa_data, epsilon_data, inter_kappa, inter_epsilon, folder):
    fig_4_B_b, ax_4_B_b = new_figure()
    fig_4_stat, ax_4_stat = new_figure()

    ax_4_B_b.plot(t, B, color='blue', lw=PLOT_STYLE["linewidth"], label=r"$B(t)$")
    ax_4_B_b.plot(t, b, color='orange', lw=PLOT_STYLE["linewidth"], label=r"$b(t)$")
    if inter_kappa:
        ax_4_stat.plot(t, kappa_data, color='red', lw=PLOT_STYLE["linewidth"], label=r"$\kappa(t)$")
    if inter_epsilon:
        ax_4_stat.plot(t, epsilon_data, color='green', lw=PLOT_STYLE["linewidth"], label=r"$\varepsilon(t)$")

    if inter_epsilon and not inter_kappa:
        stat_ylabel = "Dynamo Number"
    elif inter_kappa and not inter_epsilon:
        stat_ylabel = "Coupling factor"
    else:
        stat_ylabel = "Stochastic parameters"

    style_axis(ax_4_B_b, ylabel="Magnetic Amplitudes")
    style_axis(ax_4_stat, ylabel=stat_ylabel)

    fig_4_B_b.tight_layout()
    fig_4_stat.tight_layout()
    save_fig(fig_4_B_b, folder, "Fig_4_B_b")
    save_fig(fig_4_stat, folder, "Fig_4_stat")

def big_simus():
    tqdm.write("Simulations d'intermittence : début")
    for Lambda in tqdm(Lambdas, desc="Lambda", position=0, leave=True):
        Lambda_folder = os.path.join(datadir_mid, "Lambda=" + str(Lambda))
        for epsiloneq in tqdm(epsiloneqs, desc="epsiloneqs", position=1, leave=False):
            epsiloneq_folder = os.path.join(Lambda_folder, "epsiloneq=" + str(epsiloneq))

            for (inter_kappa, inter_epsilon) in inter_list:

                if (not inter_kappa) and inter_epsilon:
                    folder_fig_3 = os.path.join(epsiloneq_folder, "kappa_dependency")
                    fig_3_B, ax_3_B = new_figure()
                    fig_3_b, ax_3_b = new_figure()
                    minimas_folder = os.path.join(folder_fig_3, "minimas")
                    for run_idx, kappaeq in enumerate(tqdm(kappaeqs, desc="kappa_dependency", position=2, leave=False)):
                        cfg = config(datadir=folder_fig_3, term='mid',
                                     epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq,
                                     run_index=run_idx + 1)
                        (B_eq, b_eq) = cfg.get_eq()[0]
                        cfg.B0 = B_eq
                        cfg.b0 = b_eq
                        data_list, data_avg = cfg.run_and_avg(save_all=True)
                        minimas_total = []
                        for data in data_list:
                            minimas = cfg.stat_analysis(data)
                            cfg.write_stat_file(minimas=minimas)
                            minimas_total.extend(minimas)
                        cfg.plot_histograms(minimas_list=minimas_total, differentfolder=minimas_folder,
                                             name="kappaeq=" + str(kappaeq) + ".png")
                        cfg.plot_histograms(minimas_list=minimas_total, differentfolder=minimas_folder,
                                             name="kappaeq=" + str(kappaeq) + ".png")
                        cfg.plot_time(data=data_avg, type='epsilon', differentfolder=folder_fig_3, name="stat_kappa="+str(kappaeq))
                        t = data_avg[:, 0]
                        B = data_avg[:, 1]
                        b = data_avg[:, 2]
                        
                        ax_3_B.plot(t, B, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))
                        ax_3_b.plot(t, b, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))

                    for ax in [ax_3_B, ax_3_b]:
                        style_axis(ax, xlabel="t")

                    fig_3_B.tight_layout()
                    fig_3_b.tight_layout()

                    save_fig(fig_3_B, folder_fig_3, "Fig_3_B_large", dpi=300)
                    save_fig(fig_3_b, folder_fig_3, "Fig_3_b_small", dpi=300)
                    cfg.write_config_file()

                if inter_kappa and (not inter_epsilon):
                    skumanich_folder = os.path.join(epsiloneq_folder, "skumanich")
                    for kappaeq in tqdm(kappaeqs, desc="skumanich", position=2, leave=False):
                        kappaeq_folder = os.path.join(skumanich_folder, "kappaeq=" + str(kappaeq))
                        cfg = config(datadir=kappaeq_folder, term='long',
                                     epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, run_index=1)
                        (B_eq, b_eq) = cfg.get_eq(skuma=True)[0]
                        cfg.B0 = B_eq
                        cfg.b0 = b_eq
                        _, data_avg = cfg.run_and_avg(save_all=True, save_figs=True)
                        for type in ['Bb', 'kappa', 'epsilon', 'Omega']:
                            cfg.plot_time(data_avg, type=type, show=False)

                if inter_kappa and inter_epsilon:
                    comportements_folder = os.path.join(epsiloneq_folder, "comportements_divers")
                    for kappaeq in tqdm(kappaeqs, desc="intermittency", position=2, leave=False):
                        kappaeq_folder = os.path.join(comportements_folder, "kappaeq=" + str(kappaeq))
                        minimas_list = []
                        cfg = None
                        for i in range(2):
                            cfg = config(datadir=kappaeq_folder, term='mid',
                                         epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, run_index=i + 1)
                            (B_eq, b_eq) = cfg.get_eq()[0]
                            cfg.B0 = B_eq
                            cfg.b0 = b_eq
                            data = cfg.run(save=True)
                            t = data[:, 0]
                            B = data[:, 1]
                            b = data[:, 2]
                            kappa_data = data[:, 3]
                            epsilon_data = data[:, 4]
                            minimas = cfg.stat_analysis(data=data)
                            minimas_list.extend(minimas)
                            plot_fft(t, kappa_data, r"$\kappa(t)$", cfg.folder, "kappa")
                            plot_fig5(epsilon_data, kappa_data, B, cfg.folder)
                            plot_fig4(t, B, b, kappa_data, epsilon_data, inter_kappa, inter_epsilon, cfg.folder)
                        cfg.plot_histograms(minimas_list=minimas_list, differentfolder=kappaeq_folder)
    tqdm.write("Simulations d'intermittence : fin")

if __name__ == "__main__":
    #court_terme()
    big_simus()