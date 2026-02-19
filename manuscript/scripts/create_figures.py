#!/usr/bin/env python3
"""
Generate publication-quality figures for the MAGICC Nature Methods manuscript.

Generates figures as both PNG (300 DPI) and PDF:
  Figure 1: Motivating analysis (existing tools' limitations) -- 5 panels (a-e)
  Figure 2: MAGICC workflow (complete pipeline schematic) -- 1 panel
  Figure 3: Benchmark performance (MAGICC vs competitors) -- 7 panels (a-g)
  Figures S1-S5: Per-set predicted vs true quality scores (supplementary)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = '/home/tianrm/projects/magicc2/data/benchmarks'
MOTIV = os.path.join(BASE, 'motivating_v2')
OUT_DIR = '/home/tianrm/projects/magicc2/manuscript/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Verify all required data files exist
# ---------------------------------------------------------------------------
def verify_data_files():
    """Check that all required prediction TSV files exist."""
    required = []
    # Motivating sets (no MAGICC)
    for s in ['set_A', 'set_B', 'set_C']:
        for tool in ['checkm2', 'cocopye', 'deepcheck']:
            required.append(os.path.join(MOTIV, s, f'{tool}_predictions.tsv'))
    # Benchmark sets (all 4 tools)
    for s in ['set_A_v2', 'set_B_v2', 'set_C', 'set_D', 'set_E']:
        for tool in ['magicc', 'checkm2', 'cocopye', 'deepcheck']:
            required.append(os.path.join(BASE, s, f'{tool}_predictions.tsv'))
    missing = [f for f in required if not os.path.isfile(f)]
    if missing:
        print("ERROR: Missing required data files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)
    print(f"All {len(required)} required data files verified.")

# ---------------------------------------------------------------------------
# Style configuration -- Nature Methods
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,  # TrueType fonts in PDF
    'ps.fonttype': 42,
})

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
# Figure 1: 3-tool motivating palette (clearly distinguishable)
MOTIV_COLORS = {
    'CheckM2':   '#1f77b4',   # blue
    'CoCoPyE':   '#1b9e77',   # teal
    'DeepCheck': '#7570b3',   # blue-purple
}

# Figure 3: 4-tool benchmark palette
BENCH_COLORS = {
    'MAGICC':    '#d62728',   # red
    'CheckM2':   '#1f77b4',   # blue
    'CoCoPyE':   '#2ca02c',   # green
    'DeepCheck': '#9467bd',   # purple
}


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=9):
    """Add bold lowercase panel label (Nature Methods style)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    png_path = os.path.join(OUT_DIR, f'{name}.png')
    pdf_path = os.path.join(OUT_DIR, f'{name}.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'  Saved {name}.png and {name}.pdf')


def load_predictions(set_dir, tool):
    """Load a prediction TSV file."""
    path = os.path.join(set_dir, f'{tool}_predictions.tsv')
    return pd.read_csv(path, sep='\t')


def compute_mae(df, metric='completeness'):
    """Compute MAE for a given metric from a predictions dataframe."""
    return np.mean(np.abs(df[f'pred_{metric}'] - df[f'true_{metric}']))


# =========================================================================
# Helper: Draw Figure 1 panel a schematic (black/grey only, reorganized)
# =========================================================================
def _draw_figure1_panel_a(ax):
    """Draw the motivating sets schematic on the given axes (black/grey only)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis('off')

    bw = 2.8
    bh = 2.2
    by = 0.6
    positions = [(0.3, by), (3.6, by), (6.9, by)]
    set_info = [
        ('Set A (completeness gradient)', 'comp: 50-100%  cont: 0%', '1,000'),
        ('Set B (contamination gradient)', 'comp: 100%  cont: 0-80%', '1,000'),
        ('Set C (realistic mix)', 'comp: 50-100%  cont: 0-100%', '1,000'),
    ]

    for (x, y), (title, desc, n) in zip(positions, set_info):
        # Box with light grey fill
        rect = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.1",
                              facecolor='#f0f0f0', edgecolor='black', linewidth=0.7)
        ax.add_patch(rect)
        # Title at top (font size +20%: 6 -> 7.2)
        ax.text(x + bw/2, y + bh - 0.15, title, ha='center', va='top',
                fontsize=7.2, fontweight='bold', color='black')
        # Description line (font size +20%: 5 -> 6)
        ax.text(x + bw/2, y + bh - 0.50, desc,
                ha='center', va='top', fontsize=6, color='#333333')

    # Visual elements in the middle zone of each box
    gradient_y = by + 0.55
    visual_h = 0.40

    # Set A: completeness gradient boxes (black shades) with labels above
    set_a_labels = [50, 60, 70, 80, 90, 100]
    for i, (alpha, label) in enumerate(zip([0.15, 0.30, 0.45, 0.60, 0.80, 1.0], set_a_labels)):
        rx = 0.6 + i * 0.38
        rect = Rectangle((rx, gradient_y), 0.34, visual_h,
                         facecolor='black', alpha=alpha, edgecolor='#999999',
                         linewidth=0.3)
        ax.add_patch(rect)
        # Label above box
        ax.text(rx + 0.17, gradient_y + visual_h + 0.06, str(label),
                ha='center', va='bottom', fontsize=5, color='#333333')

    # Set B: contamination gradient boxes (grey to black) with labels above
    set_b_labels = [0, 20, 40, 60, 80]
    for i, (alpha, label) in enumerate(zip([0.0, 0.25, 0.5, 0.75, 1.0], set_b_labels)):
        rx = 3.9 + i * 0.44
        rect = Rectangle((rx, gradient_y), 0.40, visual_h,
                         facecolor='black', alpha=max(0.05, alpha),
                         edgecolor='#999999', linewidth=0.3)
        ax.add_patch(rect)
        # Label above box
        ax.text(rx + 0.20, gradient_y + visual_h + 0.06, str(label),
                ha='center', va='bottom', fontsize=5, color='#333333')

    # Set C: mixed scatter dots (grey/black)
    rng = np.random.RandomState(42)
    for _ in range(30):
        cx = 7.2 + rng.uniform(0, 2.2)
        cy = gradient_y + rng.uniform(0, visual_h)
        shade = rng.choice(['black', '#666666', '#999999'])
        ax.plot(cx, cy, 'o', markersize=2, color=shade, alpha=0.5)

    # n = labels at bottom of each box (font size +20%: 4.5 -> 5.4)
    for (x, y), (_, _, n) in zip(positions, set_info):
        ax.text(x + bw/2, y + 0.10, f'n = {n}', ha='center', va='bottom',
                fontsize=5.4, color='#555555', style='italic')


# =========================================================================
# Helper: Draw Figure 3 panel a schematic (same style as Figure 1 panel a)
# -- black frames, gradient boxes in A/B, scatter in E, descriptive titles
# =========================================================================
def _draw_figure3_panel_a(ax):
    """Draw the benchmark sets schematic matching Figure 1a style."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis('off')

    set_info = [
        ('Set A\n(completeness gradient)', 'comp: 50-100%\ncont: 0%', '1,000'),
        ('Set B\n(contamination gradient)', 'comp: 100%\ncont: 0-80%', '1,000'),
        ('Set C\n(Patescibacteria)', 'mixed', '1,000'),
        ('Set D\n(Archaea)', 'mixed', '1,000'),
        ('Set E\n(realistic mix)', 'comp: 50-100%\ncont: 0-100%', '1,000'),
    ]

    box_w = 1.72
    box_h = 1.8
    gap = 0.30
    total_w = 5 * box_w + 4 * gap
    start_x = (10 - total_w) / 2
    by = 0.3
    positions = [start_x + i * (box_w + gap) for i in range(5)]

    for i, (title, desc, n) in enumerate(set_info):
        bx = positions[i]
        # Box with black frame (same style as fig 1a)
        rect = FancyBboxPatch((bx, by), box_w, box_h,
                              boxstyle="round,pad=0.08",
                              facecolor='#f0f0f0', edgecolor='black',
                              linewidth=0.7)
        ax.add_patch(rect)
        # Title (reduced font size to fit narrower boxes)
        ax.text(bx + box_w/2, by + box_h - 0.06, title,
                ha='center', va='top', fontsize=5.8, fontweight='bold',
                color='black')
        # Description
        ax.text(bx + box_w/2, by + box_h - 0.55, desc,
                ha='center', va='top', fontsize=5, color='#333333')
        # n = label
        ax.text(bx + box_w/2, by + 0.06, f'n = {n}',
                ha='center', va='bottom', fontsize=5, color='#555555',
                style='italic')

    # Visual elements in the middle zone of boxes
    gradient_y = by + 0.35
    visual_h = 0.32
    margin_inner = 0.12
    inner_w = box_w - 2 * margin_inner

    # Set A: completeness gradient boxes (like fig 1a)
    bx_a = positions[0]
    for j, alpha in enumerate([0.15, 0.30, 0.45, 0.60, 0.80, 1.0]):
        rw = inner_w / 6
        rx = bx_a + margin_inner + j * rw
        rect = Rectangle((rx, gradient_y), rw * 0.88, visual_h,
                         facecolor='black', alpha=alpha, edgecolor='#999999',
                         linewidth=0.3)
        ax.add_patch(rect)

    # Set B: contamination gradient boxes (like fig 1a)
    bx_b = positions[1]
    for j, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        rw = inner_w / 5
        rx = bx_b + margin_inner + j * rw
        rect = Rectangle((rx, gradient_y), rw * 0.88, visual_h,
                         facecolor='black', alpha=max(0.05, alpha),
                         edgecolor='#999999', linewidth=0.3)
        ax.add_patch(rect)

    # Set E: scatter dots (like fig 1a Set C)
    bx_e = positions[4]
    rng = np.random.RandomState(42)
    for _ in range(25):
        cx = bx_e + margin_inner + rng.uniform(0, inner_w)
        cy = gradient_y + rng.uniform(0, visual_h)
        shade = rng.choice(['black', '#666666', '#999999'])
        ax.plot(cx, cy, 'o', markersize=1.5, color=shade, alpha=0.5)


# =========================================================================
# FIGURE 1: Motivating Analysis (5 panels: a-e, 3 rows)
# -- All font sizes increased by 20%, bar values rounded to 1 decimal
# =========================================================================
def make_figure1():
    print('Creating Figure 1: Motivating Analysis ...')

    tools_motiv = ['checkm2', 'cocopye', 'deepcheck']
    tool_labels = {'checkm2': 'CheckM2', 'cocopye': 'CoCoPyE', 'deepcheck': 'DeepCheck'}
    tool_colors = {t: MOTIV_COLORS[tool_labels[t]] for t in tools_motiv}

    set_dirs = {
        'A': os.path.join(MOTIV, 'set_A'),
        'B': os.path.join(MOTIV, 'set_B'),
        'C': os.path.join(MOTIV, 'set_C'),
    }

    # Pre-load all data
    data = {}
    for sname, sdir in set_dirs.items():
        data[sname] = {}
        for tool in tools_motiv:
            data[sname][tool] = load_predictions(sdir, tool)

    # Compute MAEs dynamically
    comp_maes = {}
    cont_maes = {}
    for sname in ['A', 'B', 'C']:
        comp_maes[sname] = {}
        cont_maes[sname] = {}
        for tool in tools_motiv:
            df = data[sname][tool]
            comp_maes[sname][tool] = compute_mae(df, 'completeness')
            cont_maes[sname][tool] = compute_mae(df, 'contamination')

    # New layout: 3 rows
    # Row 1: panel a (full width, reduced height)
    # Row 2: panels b, c (2 columns)
    # Row 3: panels d, e (2 columns)
    fig = plt.figure(figsize=(7.5, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.0, 1.0],
                          hspace=0.45, wspace=0.38,
                          left=0.09, right=0.97, top=0.95, bottom=0.07)

    # ------------------------------------------------------------------
    # Panel a: Cartoon/schematic of the 3 motivating sets (full width)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    add_panel_label(ax_a, 'a', x=-0.03, y=1.15, fontsize=10.8)
    _draw_figure1_panel_a(ax_a)

    # ------------------------------------------------------------------
    # Panel b: Completeness MAE bar plot (sets A, B, C)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, 'b', x=-0.14, y=1.08, fontsize=10.8)

    set_names = ['A', 'B', 'C']
    n_sets = len(set_names)
    width = 0.22
    x = np.arange(n_sets)

    for ti, tool in enumerate(tools_motiv):
        maes = [comp_maes[s][tool] for s in set_names]
        offset = (ti - 1) * width
        bars = ax.bar(x + offset, maes, width,
                      color=tool_colors[tool], label=tool_labels[tool],
                      edgecolor='none', alpha=0.8)
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=5.4,
                    fontweight='bold', color=tool_colors[tool])

    ax.set_xticks(x)
    ax.set_xticklabels([f'Set {s}' for s in set_names], fontsize=7.2)
    ax.set_ylabel('Completeness MAE (%)', fontsize=8.4)
    ax.set_title('Completeness MAE', fontsize=8.4, pad=3)
    ax.legend(frameon=False, fontsize=6.6, loc='upper left')
    ax.tick_params(labelsize=7.2)

    # No red arrow annotation -- removed
    max_c = max(max(comp_maes[s][t] for t in tools_motiv) for s in set_names)
    ax.set_ylim(0, max_c + 3)

    # ------------------------------------------------------------------
    # Panel c: Contamination MAE bar plot (sets A, B, C)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, 'c', x=-0.14, y=1.08, fontsize=10.8)

    for ti, tool in enumerate(tools_motiv):
        maes = [cont_maes[s][tool] for s in set_names]
        offset = (ti - 1) * width
        bars = ax.bar(x + offset, maes, width,
                      color=tool_colors[tool], label=tool_labels[tool],
                      edgecolor='none', alpha=0.8)
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=5.4,
                    fontweight='bold', color=tool_colors[tool])

    ax.set_xticks(x)
    ax.set_xticklabels([f'Set {s}' for s in set_names], fontsize=7.2)
    ax.set_ylabel('Contamination MAE (%)', fontsize=8.4)
    ax.set_title('Contamination MAE', fontsize=8.4, pad=3)
    ax.legend(frameon=False, fontsize=6.6, loc='upper left')
    ax.tick_params(labelsize=7.2)

    max_bc = max(max(cont_maes[s][t] for t in tools_motiv) for s in set_names)
    ax.set_ylim(0, max_bc + 5)

    # ------------------------------------------------------------------
    # Panel d (NEW): Scatter predicted vs true contamination for Set B
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    add_panel_label(ax, 'd', x=-0.14, y=1.08, fontsize=10.8)

    scatter_colors = {
        'checkm2':   MOTIV_COLORS['CheckM2'],
        'cocopye':   MOTIV_COLORS['CoCoPyE'],
        'deepcheck': MOTIV_COLORS['DeepCheck'],
    }

    for tool in tools_motiv:
        df = data['B'][tool]
        ax.scatter(df['true_contamination'], df['pred_contamination'],
                   s=8, alpha=0.35, color=scatter_colors[tool],
                   label=tool_labels[tool], edgecolors='none', rasterized=True)

    ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=0.8, zorder=0)

    ax.set_xlabel('True contamination (%)', fontsize=8.4)
    ax.set_ylabel('Predicted contamination (%)', fontsize=8.4)
    ax.set_title('Pred. vs. true cont. (Set B)', fontsize=8.4, pad=3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=7.2)

    # No MAE text annotations
    ax.legend(frameon=False, fontsize=6.6, loc='upper left')

    # ------------------------------------------------------------------
    # Panel e (was panel d): Scatter predicted vs true contamination for Set C
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    add_panel_label(ax, 'e', x=-0.14, y=1.08, fontsize=10.8)

    for tool in tools_motiv:
        df = data['C'][tool]
        ax.scatter(df['true_contamination'], df['pred_contamination'],
                   s=8, alpha=0.35, color=scatter_colors[tool],
                   label=tool_labels[tool], edgecolors='none', rasterized=True)

    ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=0.8, zorder=0)

    ax.set_xlabel('True contamination (%)', fontsize=8.4)
    ax.set_ylabel('Predicted contamination (%)', fontsize=8.4)
    ax.set_title('Pred. vs. true cont. (Set C)', fontsize=8.4, pad=3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=7.2)

    # No MAE text annotations
    ax.legend(frameon=False, fontsize=6.6, loc='upper left')

    save_fig(fig, 'figure1_motivating')

    # ------------------------------------------------------------------
    # Save panel a as separate SVG
    # ------------------------------------------------------------------
    fig_svg = plt.figure(figsize=(7.5, 1.5))
    ax_svg = fig_svg.add_subplot(111)
    _draw_figure1_panel_a(ax_svg)
    svg_path = os.path.join(OUT_DIR, 'figure1a_schematic.svg')
    fig_svg.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_svg)
    print(f'  Saved figure1a_schematic.svg')


# =========================================================================
# FIGURE 2: MAGICC Workflow (1 panel, full width -- improved version)
# -- GTDB r226, aligned boxes, wrapped text to fit inside boxes
# =========================================================================
def make_figure2():
    print('Creating Figure 2: MAGICC Workflow ...')

    # Wide canvas for 5-phase horizontal layout
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-1.2, 6.2)
    ax.axis('off')

    # ---- Color palette (pastel, consistent per phase) ----
    c_p1  = '#C8E6C9'   # green  - Data Curation
    c_p2  = '#BBDEFB'   # blue   - K-mer Selection
    c_p3  = '#FFF9C4'   # yellow - Training Data Synthesis
    c_p4  = '#E1BEE7'   # purple - Feature Extraction
    c_p5  = '#FFCCBC'   # orange - Neural Network
    c_out = '#F8BBD0'   # pink   - Output
    c_bg  = '#FAFAFA'   # phase background

    _acc = {'p1': '#66BB6A', 'p2': '#42A5F5', 'p3': '#FFB300',
            'p4': '#AB47BC', 'p5': '#E64A19'}

    def _box(x, y, w, h, text, color, fontsize=5.5, bold=False,
             text_color='#333333', edge_color='#9E9E9E', linewidth=0.6,
             linespacing=1.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor=edge_color,
                             linewidth=linewidth)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=text_color,
                linespacing=linespacing, multialignment='center')

    def _arr(x1, y1, x2, y2, color='#555555', lw=0.8):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    def _xarr(x1, y1, x2, y2):
        _arr(x1, y1, x2, y2, color='#BDBDBD', lw=0.7)

    def _phase_bg(x, y, w, h, label):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=c_bg, edgecolor='#E0E0E0',
                              linewidth=0.4, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.1, label, ha='center', va='top',
                fontsize=5.5, fontweight='bold', color='#757575')

    # Column positions
    PX = [0.0, 3.6, 7.2, 10.5, 14.0]
    PW = [3.3, 3.3, 3.0, 3.2, 4.2]
    BW = [2.8, 2.8, 2.5, 2.7, 3.6]
    BX = [PX[i] + (PW[i] - BW[i])/2 for i in range(5)]
    ROW = [4.2, 2.8, 1.4, 0.0]
    BH = 0.85

    # Phase backgrounds
    _bg_y, _bg_h = -0.8, 6.6
    for i, label in enumerate([
        'Phase 1\nData Curation', 'Phase 2\nK-mer Selection',
        'Phase 3\nTraining Synthesis', 'Phase 4\nFeature Extraction',
        'Phase 5\nNeural Network',
    ]):
        _phase_bg(PX[i], _bg_y, PW[i], _bg_h, label)

    # --- Phase 1: Data Curation ---
    _box(BX[0], ROW[0], BW[0], BH, 'GTDB r226\n732K genomes', c_p1, 6, True)
    _box(BX[0], ROW[1], BW[0], BH,
         'Quality filters\ncomp >98%, cont <2%\n<100 contigs\nN50 >20 kbp', c_p1, 4.5)
    _box(BX[0], ROW[2], BW[0], BH, '277K HQ genomes', c_p1, 5.5, True)
    _box(BX[0], ROW[3], BW[0], BH,
         '100K selected\n(stratified sampling,\n110 phyla)',
         c_p1, 5, True, edge_color=_acc['p1'], linewidth=0.9)
    cx1 = BX[0] + BW[0]/2
    for r in range(3):
        _arr(cx1, ROW[r], cx1, ROW[r+1] + BH)

    # --- Phase 2: K-mer Feature Selection ---
    _box(BX[1], ROW[0], BW[1], BH,
         '2,000 representatives\n(1K bact. + 1K arch.)', c_p2, 5)
    _box(BX[1], ROW[1], BW[1], BH,
         'Core gene identification\nProdigal + HMMER\n(85 bact.+128 arch.\nHMMs)', c_p2, 4.5)
    _box(BX[1], ROW[2], BW[1], BH, '9-mer counting\n(KMC3)', c_p2, 5.5)
    _box(BX[1], ROW[3], BW[1], BH,
         'Top 9,249 canonical\nk-mers (by prevalence)',
         c_p2, 4, True, edge_color=_acc['p2'], linewidth=0.9)
    cx2 = BX[1] + BW[1]/2
    for r in range(3):
        _arr(cx2, ROW[r], cx2, ROW[r+1] + BH)
    _xarr(BX[0] + BW[0], ROW[0] + BH/2, BX[1], ROW[0] + BH/2)

    # --- Phase 3: Training Data Synthesis (4 aligned boxes) ---
    _box(BX[2], ROW[0], BW[2], BH,
         '100K ref. genomes\nFragmentation\n(4 quality tiers)', c_p3, 4.8)
    _box(BX[2], ROW[1], BW[2], BH,
         'Contamination injection\nwithin + cross-phylum\n(Dirichlet allocation)', c_p3, 4.5)
    _box(BX[2], ROW[2], BW[2], BH,
         '1M synthetic\ngenomes', c_p3, 5, True)
    _box(BX[2], ROW[3], BW[2], BH,
         '800K train\n100K validation\n100K test',
         c_p3, 5, True, edge_color=_acc['p3'], linewidth=0.9)
    cx3 = BX[2] + BW[2]/2
    for r in range(3):
        _arr(cx3, ROW[r], cx3, ROW[r+1] + BH)
    _xarr(BX[0] + BW[0], ROW[3] + BH/2, BX[2], ROW[0] + BH/2)

    # --- Phase 4: Feature Extraction (4 aligned boxes) ---
    _box(BX[3], ROW[0], BW[3], BH, 'Input FASTA', c_p4, 6, True)
    _box(BX[3], ROW[1], BW[3], BH,
         'K-mer counting\n9,249 canonical 9-mers\n(Numba rolling hash)', c_p4, 4.5)
    _box(BX[3], ROW[2], BW[3], BH,
         'Assembly statistics\n26 features (contigs,\nGC, k-mer stats)', c_p4, 4.5)
    _box(BX[3], ROW[3], BW[3], BH,
         'Normalization\n(Z-score, log,\nmin-max, robust)',
         c_p4, 4.5, edge_color=_acc['p4'], linewidth=0.9)
    cx4 = BX[3] + BW[3]/2
    # Input -> K-mer counting
    _arr(cx4, ROW[0], cx4, ROW[1] + BH)
    # Input -> Assembly stats (bypass arrow on the right)
    _arr(cx4 + BW[3]/2 - 0.15, ROW[0], cx4 + BW[3]/2 - 0.15, ROW[2] + BH)
    # K-mer -> Normalization (left side)
    _arr(cx4 - 0.4, ROW[1], cx4 - 0.4, ROW[3] + BH)
    # Assembly -> Normalization (right side)
    _arr(cx4 + 0.4, ROW[2], cx4 + 0.4, ROW[3] + BH)
    ax.text(cx4, ROW[0] + BH + 0.12, '(also inference pipeline)',
            ha='center', va='bottom', fontsize=4, color='#7B1FA2', style='italic')
    _xarr(BX[1] + BW[1], ROW[3] + BH/2, BX[3], ROW[1] + BH/2)
    _xarr(BX[2] + BW[2], ROW[3] + BH/2, BX[3], ROW[3] + BH/2)

    # --- Phase 5: Neural Network Model (4 aligned boxes) ---
    _box(BX[4], ROW[0], BW[4], BH,
         'K-mer branch\n9,249\u21924,096\u21921,024\u2192256\n(FC + SE attention blocks)',
         c_p5, 4.8)
    _box(BX[4], ROW[1], BW[4], BH,
         'Cross-attention fusion\n(assembly features query\nk-mer representations)',
         c_p5, 4.8, True, edge_color=_acc['p5'], linewidth=0.9)
    _box(BX[4], ROW[2], BW[4], BH,
         'Assembly branch\n26 \u2192 128 \u2192 64\n(FC layers)', c_p5, 5)
    _box(BX[4], ROW[3], BW[4], BH,
         'Prediction head\nCompleteness [50\u2013100%]\nContamination [0\u2013100%]',
         c_out, 5.5, True, text_color='#C62828', edge_color='#E53935', linewidth=1.0)
    cx5 = BX[4] + BW[4]/2
    _arr(cx5, ROW[0], cx5, ROW[1] + BH)
    _arr(cx5, ROW[2] + BH, cx5, ROW[1])
    _arr(cx5, ROW[1], cx5, ROW[3] + BH)
    _xarr(BX[3] + BW[3], ROW[1] + BH/2, BX[4], ROW[0] + BH/2)
    _xarr(BX[3] + BW[3], ROW[2] + BH/2, BX[4], ROW[2] + BH/2)

    save_fig(fig, 'figure2_workflow')


# =========================================================================
# FIGURE 3: Benchmark Performance (7 panels: a-g, 3 rows)
# -- Same style as Figure 1: font sizes +20%, no bar values, 1-col legend,
#    shorter panel a with fig-1a style, scatter s=6 (original)
# =========================================================================
def make_figure3():
    print('Creating Figure 3: Benchmark Performance ...')

    bench_sets = {
        'A': 'set_A_v2',
        'B': 'set_B_v2',
        'C': 'set_C',
        'D': 'set_D',
        'E': 'set_E',
    }
    set_labels_ordered = ['A', 'B', 'C', 'D', 'E']
    tools = ['magicc', 'checkm2', 'cocopye', 'deepcheck']
    tool_labels = {
        'magicc': 'MAGICC', 'checkm2': 'CheckM2',
        'cocopye': 'CoCoPyE', 'deepcheck': 'DeepCheck'
    }
    tool_colors = {t: BENCH_COLORS[tool_labels[t]] for t in tools}

    # Load all data
    all_data = {}
    for slabel, sdir_name in bench_sets.items():
        all_data[slabel] = {}
        for tool in tools:
            df = load_predictions(os.path.join(BASE, sdir_name), tool)
            all_data[slabel][tool] = df

    # Compute MAEs dynamically
    comp_maes = {}
    cont_maes = {}
    for slabel in set_labels_ordered:
        comp_maes[slabel] = {}
        cont_maes[slabel] = {}
        for tool in tools:
            df = all_data[slabel][tool]
            comp_maes[slabel][tool] = compute_mae(df, 'completeness')
            cont_maes[slabel][tool] = compute_mae(df, 'contamination')

    # Layout: 3 rows -- panel a shorter height
    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.38, 1.0, 1.0],
                          hspace=0.30, wspace=0.40,
                          left=0.07, right=0.97, top=0.96, bottom=0.06)

    # ------------------------------------------------------------------
    # Panel a: Cartoon showing profiles of benchmark sets A-E (full width)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    add_panel_label(ax_a, 'a', x=-0.02, y=1.18, fontsize=10.8)
    _draw_figure3_panel_a(ax_a)

    # ------------------------------------------------------------------
    # Panel b: Completeness MAE bar plot (all 5 sets, 4 tools) -- 2 cols
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    add_panel_label(ax, 'b', x=-0.08, y=1.08, fontsize=10.8)

    n_sets = len(set_labels_ordered)
    n_tools = len(tools)
    width = 0.17
    x = np.arange(n_sets)

    for ti, tool in enumerate(tools):
        maes = [comp_maes[s][tool] for s in set_labels_ordered]
        offset = (ti - 1.5) * width
        bars = ax.bar(x + offset, maes, width,
                      color=tool_colors[tool], label=tool_labels[tool],
                      edgecolor='none', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Set {s}' for s in set_labels_ordered], fontsize=7.2)
    ax.set_ylabel('Completeness MAE (%)', fontsize=8.4)
    ax.set_title('Completeness MAE by set', fontsize=8.4, pad=3)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc='upper left')
    ax.tick_params(labelsize=7.2)

    # ------------------------------------------------------------------
    # Panel c: Contamination MAE bar plot (all 5 sets, 4 tools) -- 2 cols
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2:4])
    add_panel_label(ax, 'c', x=-0.08, y=1.08, fontsize=10.8)

    for ti, tool in enumerate(tools):
        maes = [cont_maes[s][tool] for s in set_labels_ordered]
        offset = (ti - 1.5) * width
        bars = ax.bar(x + offset, maes, width,
                      color=tool_colors[tool], label=tool_labels[tool],
                      edgecolor='none', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Set {s}' for s in set_labels_ordered], fontsize=7.2)
    ax.set_ylabel('Contamination MAE (%)', fontsize=8.4)
    ax.set_title('Contamination MAE by set', fontsize=8.4, pad=3)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc='upper left')
    ax.tick_params(labelsize=7.2)

    # ------------------------------------------------------------------
    # Row 3: Scatter plots (4 columns: d, e, f, g) -- s=6 (original size)
    # ------------------------------------------------------------------
    plot_order = ['deepcheck', 'cocopye', 'checkm2', 'magicc']
    scatter_sets = [
        ('d', 'B', 'Pred. vs. true cont. (Set B)'),
        ('e', 'C', 'Pred. vs. true cont. (Set C)'),
        ('f', 'D', 'Pred. vs. true cont. (Set D)'),
        ('g', 'E', 'Pred. vs. true cont. (Set E)'),
    ]

    for col_idx, (panel_label, slabel, title) in enumerate(scatter_sets):
        ax = fig.add_subplot(gs[2, col_idx])
        add_panel_label(ax, panel_label, x=-0.18, y=1.08, fontsize=10.8)

        for tool in plot_order:
            df = all_data[slabel][tool]
            ax.scatter(df['true_contamination'], df['pred_contamination'],
                       s=6, alpha=0.35, color=tool_colors[tool],
                       label=tool_labels[tool], edgecolors='none', rasterized=True)

        ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=0.8, zorder=0)

        ax.set_xlabel('True cont. (%)', fontsize=6.6)
        ax.set_ylabel('Pred. cont. (%)', fontsize=6.6)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.tick_params(labelsize=6)

        # Legend at top left
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index('MAGICC'), labels.index('CheckM2'),
                 labels.index('CoCoPyE'), labels.index('DeepCheck')]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=False, fontsize=5.4, loc='upper left', markerscale=1.2)

    save_fig(fig, 'figure3_benchmark')

    # ------------------------------------------------------------------
    # Save panel a as separate SVG
    # ------------------------------------------------------------------
    fig_svg = plt.figure(figsize=(7.5, 1.5))
    ax_svg = fig_svg.add_subplot(111)
    _draw_figure3_panel_a(ax_svg)
    svg_path = os.path.join(OUT_DIR, 'figure3a_schematic.svg')
    fig_svg.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_svg)
    print(f'  Saved figure3a_schematic.svg')


# =========================================================================
# SUPPLEMENTARY FIGURES S1-S5: Per-set pred vs true for all 4 tools
# =========================================================================
def make_supplementary_figures():
    print('Creating Supplementary Figures S1-S5 ...')

    bench_sets_ordered = [
        ('A', 'set_A_v2', 'Set A (completeness gradient)'),
        ('B', 'set_B_v2', 'Set B (contamination gradient)'),
        ('C', 'set_C',    'Set C (Patescibacteria)'),
        ('D', 'set_D',    'Set D (Archaea)'),
        ('E', 'set_E',    'Set E (realistic mixed)'),
    ]
    tools = ['magicc', 'checkm2', 'cocopye', 'deepcheck']
    tool_labels = {
        'magicc': 'MAGICC', 'checkm2': 'CheckM2',
        'cocopye': 'CoCoPyE', 'deepcheck': 'DeepCheck'
    }
    tool_colors = {t: BENCH_COLORS[tool_labels[t]] for t in tools}

    for fig_idx, (slabel, sdir_name, set_title) in enumerate(bench_sets_ordered, start=1):
        fig_name = f'figS{fig_idx}_set{slabel}'
        print(f'  Creating {fig_name} ...')

        # Load data for this set
        set_data = {}
        for tool in tools:
            set_data[tool] = load_predictions(os.path.join(BASE, sdir_name), tool)

        fig, (ax_comp, ax_cont) = plt.subplots(1, 2, figsize=(7, 3.2))
        fig.subplots_adjust(wspace=0.35, left=0.09, right=0.97, top=0.88, bottom=0.13)

        # Panel (a): Predicted vs true COMPLETENESS
        add_panel_label(ax_comp, 'a', x=-0.14, y=1.10)

        plot_order = ['deepcheck', 'cocopye', 'checkm2', 'magicc']
        for tool in plot_order:
            df = set_data[tool]
            ax_comp.scatter(df['true_completeness'], df['pred_completeness'],
                           s=8, alpha=0.35, color=tool_colors[tool],
                           label=tool_labels[tool], edgecolors='none', rasterized=True)

        ax_comp.plot([0, 100], [0, 100], '--', color='gray', linewidth=0.8, zorder=0)
        ax_comp.set_xlabel('True completeness (%)')
        ax_comp.set_ylabel('Predicted completeness (%)')
        ax_comp.set_title(f'Completeness -- {set_title}', fontsize=7, pad=3)
        ax_comp.set_xlim(-5, 105)
        ax_comp.set_ylim(-5, 105)

        # No MAE text box -- removed
        # Legend at top left
        handles, labels = ax_comp.get_legend_handles_labels()
        order = [labels.index('MAGICC'), labels.index('CheckM2'),
                 labels.index('CoCoPyE'), labels.index('DeepCheck')]
        ax_comp.legend([handles[i] for i in order], [labels[i] for i in order],
                      frameon=False, fontsize=5.5, loc='upper left', markerscale=1.5)

        # Panel (b): Predicted vs true CONTAMINATION
        add_panel_label(ax_cont, 'b', x=-0.14, y=1.10)

        for tool in plot_order:
            df = set_data[tool]
            ax_cont.scatter(df['true_contamination'], df['pred_contamination'],
                           s=8, alpha=0.35, color=tool_colors[tool],
                           label=tool_labels[tool], edgecolors='none', rasterized=True)

        ax_cont.plot([0, 100], [0, 100], '--', color='gray', linewidth=0.8, zorder=0)
        ax_cont.set_xlabel('True contamination (%)')
        ax_cont.set_ylabel('Predicted contamination (%)')
        ax_cont.set_title(f'Contamination -- {set_title}', fontsize=7, pad=3)
        ax_cont.set_xlim(-5, 105)
        ax_cont.set_ylim(-5, 105)

        # No MAE text box -- removed
        # Legend at top left
        handles, labels = ax_cont.get_legend_handles_labels()
        order = [labels.index('MAGICC'), labels.index('CheckM2'),
                 labels.index('CoCoPyE'), labels.index('DeepCheck')]
        ax_cont.legend([handles[i] for i in order], [labels[i] for i in order],
                      frameon=False, fontsize=5.5, loc='upper left', markerscale=1.5)

        save_fig(fig, fig_name)


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    print(f'Output directory: {OUT_DIR}')
    verify_data_files()
    make_figure1()
    make_figure2()
    make_figure3()
    make_supplementary_figures()
    print('\nAll figures generated successfully.')
