#!/usr/bin/env python3
"""
Generate an improved publication-quality Figure 2: MAGICC workflow diagram
for Nature Methods manuscript.

Outputs PNG (300 DPI) and PDF to manuscript/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = '/home/tianrm/projects/magicc2/manuscript/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style configuration -- Nature Methods
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def make_figure2():
    """Create Figure 2: MAGICC pipeline workflow diagram."""
    print('Creating Figure 2: MAGICC Workflow (improved) ...')

    # Wide canvas for 5-phase horizontal layout
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-1.2, 6.2)
    ax.axis('off')

    # ---- Color palette (pastel, consistent per phase) ----
    colors = {
        'p1':  '#C8E6C9',  # green  - Data Curation
        'p2':  '#BBDEFB',  # blue   - K-mer Selection
        'p3':  '#FFF9C4',  # yellow - Training Data Synthesis
        'p4':  '#E1BEE7',  # purple - Feature Extraction
        'p5':  '#FFCCBC',  # orange - Neural Network
        'out': '#F8BBD0',  # pink   - Output
        'bg':  '#FAFAFA',  # phase background
    }

    accent = {
        'p1': '#66BB6A', 'p2': '#42A5F5', 'p3': '#FFB300',
        'p4': '#AB47BC', 'p5': '#E64A19',
    }

    # ---- Helper functions ----
    def add_box(x, y, w, h, text, color, fontsize=5.5, bold=False,
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

    def arrow(x1, y1, x2, y2, color='#555555', lw=0.8, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    def cross_arrow(x1, y1, x2, y2):
        """Gray dashed arrow for cross-phase connections."""
        arrow(x1, y1, x2, y2, color='#BDBDBD', lw=0.7, style='->')

    def phase_bg(x, y, w, h, label):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=colors['bg'], edgecolor='#E0E0E0',
                              linewidth=0.4, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.1, label, ha='center', va='top',
                fontsize=5.5, fontweight='bold', color='#757575')

    # ====================================================================
    # Column positions (left edge of each phase region)
    # ====================================================================
    # 5 phases spread across x=0..18
    PX = [0.0, 3.6, 7.2, 10.5, 14.0]  # left edge of phase bg
    PW = [3.3, 3.3, 3.0, 3.2, 4.2]    # width of phase bg

    # Box widths within each phase
    BW = [2.8, 2.8, 2.5, 2.7, 3.6]

    # Box left offsets (centered in phase)
    BX = [PX[i] + (PW[i] - BW[i])/2 for i in range(5)]

    # Vertical positions for rows of boxes (top to bottom)
    ROW = [4.2, 2.8, 1.4, 0.0]  # 4 rows
    BH = 0.85  # standard box height
    BH_SM = 0.65  # small box height

    # ====================================================================
    # Phase backgrounds
    # ====================================================================
    bg_y = -0.8
    bg_h = 6.6
    phase_labels = [
        'Phase 1\nData Curation',
        'Phase 2\nK-mer Selection',
        'Phase 3\nTraining Synthesis',
        'Phase 4\nFeature Extraction',
        'Phase 5\nNeural Network',
    ]
    for i in range(5):
        phase_bg(PX[i], bg_y, PW[i], bg_h, phase_labels[i])

    # ====================================================================
    # Phase 1: Data Curation
    # ====================================================================
    i = 0
    add_box(BX[i], ROW[0], BW[i], BH,
            'GTDB r220\n732K genomes', colors['p1'], fontsize=6, bold=True)

    add_box(BX[i], ROW[1], BW[i], BH,
            'Quality filters\ncomp >98%, cont <2%\n<100 contigs, N50 >20 kbp',
            colors['p1'], fontsize=4.8)

    add_box(BX[i], ROW[2], BW[i], BH,
            '277K HQ genomes', colors['p1'], fontsize=5.5, bold=True)

    add_box(BX[i], ROW[3], BW[i], BH,
            '100K selected\n(stratified sampling, 110 phyla)',
            colors['p1'], fontsize=5, bold=True, edge_color=accent['p1'], linewidth=0.9)

    cx1 = BX[i] + BW[i]/2  # center x for phase 1
    arrow(cx1, ROW[0], cx1, ROW[1] + BH)
    arrow(cx1, ROW[1], cx1, ROW[2] + BH)
    arrow(cx1, ROW[2], cx1, ROW[3] + BH)

    # ====================================================================
    # Phase 2: K-mer Feature Selection
    # ====================================================================
    i = 1
    add_box(BX[i], ROW[0], BW[i], BH,
            '2,000 representatives\n(1K bacterial + 1K archaeal)',
            colors['p2'], fontsize=5)

    add_box(BX[i], ROW[1], BW[i], BH,
            'Core gene identification\nProdigal + HMMER\n(85 bact. + 128 arch. HMMs)',
            colors['p2'], fontsize=4.8)

    add_box(BX[i], ROW[2], BW[i], BH,
            '9-mer counting\n(KMC3)', colors['p2'], fontsize=5.5)

    add_box(BX[i], ROW[3], BW[i], BH,
            'Top 9,249 canonical\nk-mers (by prevalence)',
            colors['p2'], fontsize=5, bold=True, edge_color=accent['p2'], linewidth=0.9)

    cx2 = BX[i] + BW[i]/2
    arrow(cx2, ROW[0], cx2, ROW[1] + BH)
    arrow(cx2, ROW[1], cx2, ROW[2] + BH)
    arrow(cx2, ROW[2], cx2, ROW[3] + BH)

    # Phase 1 -> Phase 2
    cross_arrow(BX[0] + BW[0], ROW[0] + BH/2, BX[1], ROW[0] + BH/2)

    # ====================================================================
    # Phase 3: Training Data Synthesis
    # ====================================================================
    i = 2
    add_box(BX[i], ROW[0], BW[i], BH,
            '100K reference genomes\n\u2193\nFragmentation\n(4 quality tiers)',
            colors['p3'], fontsize=4.8)

    add_box(BX[i], ROW[1], BW[i], BH,
            'Contamination injection\nwithin-phylum + cross-phylum\n(Dirichlet allocation)',
            colors['p3'], fontsize=4.8)

    add_box(BX[i], ROW[2], BW[i], BH + 0.4,
            '1M synthetic genomes\n\n800K train\n100K validation\n100K test',
            colors['p3'], fontsize=5, bold=True, edge_color=accent['p3'], linewidth=0.9)

    cx3 = BX[i] + BW[i]/2
    arrow(cx3, ROW[0], cx3, ROW[1] + BH)
    arrow(cx3, ROW[1], cx3, ROW[2] + BH + 0.4)

    # Phase 1 -> Phase 3 (100K genomes feed training)
    cross_arrow(BX[0] + BW[0], ROW[3] + BH/2, BX[2], ROW[0] + BH/2)

    # ====================================================================
    # Phase 4: Feature Extraction
    # ====================================================================
    i = 3
    add_box(BX[i], ROW[0], BW[i], BH,
            'Input FASTA', colors['p4'], fontsize=6, bold=True)

    # Two parallel feature streams
    kmer_y = ROW[1] + 0.15
    asm_y = ROW[2] - 0.15
    stream_h = 1.0

    add_box(BX[i], kmer_y, BW[i], stream_h,
            'K-mer counting\n9,249 canonical 9-mers\n(Numba rolling hash)',
            colors['p4'], fontsize=4.8)

    add_box(BX[i], asm_y, BW[i], stream_h,
            'Assembly statistics\n26 features\n(contig metrics, GC, k-mer stats)',
            colors['p4'], fontsize=4.8)

    # Normalization
    add_box(BX[i], ROW[3] - 0.15, BW[i], BH_SM,
            'Normalization\n(Z-score, log, min-max, robust)',
            colors['p4'], fontsize=4.5, edge_color=accent['p4'], linewidth=0.9)

    cx4 = BX[i] + BW[i]/2
    # Input splits to two streams
    arrow(cx4, ROW[0], cx4, kmer_y + stream_h)
    # Draw a small fork
    ax.plot([cx4 - 0.3, cx4 + 0.3], [kmer_y + stream_h + 0.05, kmer_y + stream_h + 0.05],
            color='#555555', lw=0.6, solid_capstyle='round')

    # k-mer stream and assembly stream to normalization
    arrow(cx4 - 0.4, kmer_y, cx4 - 0.4, ROW[3] - 0.15 + BH_SM)
    arrow(cx4 + 0.4, asm_y, cx4 + 0.4, ROW[3] - 0.15 + BH_SM)

    # "also inference pipeline" annotation
    ax.text(cx4, ROW[0] + BH + 0.12, '(also inference pipeline)',
            ha='center', va='bottom', fontsize=4, color='#7B1FA2',
            style='italic')

    # Phase 2 -> Phase 4 (k-mer list feeds feature extraction)
    cross_arrow(BX[1] + BW[1], ROW[3] + BH/2, BX[3], kmer_y + stream_h/2)

    # Phase 3 -> Phase 4 (training data uses same feature extraction)
    cross_arrow(BX[2] + BW[2], ROW[2] + 0.5, BX[3], ROW[3] - 0.15 + BH_SM/2)

    # ====================================================================
    # Phase 5: Neural Network Model
    # ====================================================================
    i = 4
    # K-mer branch (top)
    add_box(BX[i], ROW[0], BW[i], BH,
            'K-mer branch\n9,249 \u2192 4,096 \u2192 1,024 \u2192 256\n(FC + SE attention blocks)',
            colors['p5'], fontsize=5)

    # Cross-attention fusion (middle)
    add_box(BX[i], ROW[1], BW[i], BH,
            'Cross-attention fusion\n(assembly features query\nk-mer token representations)',
            colors['p5'], fontsize=5, bold=True, edge_color=accent['p5'], linewidth=0.9)

    # Assembly branch (lower)
    add_box(BX[i], ROW[2], BW[i], BH,
            'Assembly branch\n26 \u2192 128 \u2192 64\n(FC layers)',
            colors['p5'], fontsize=5)

    # Output / prediction head
    add_box(BX[i], ROW[3] - 0.15, BW[i], BH,
            'Prediction head\nCompleteness [50\u2013100%]\nContamination [0\u2013100%]',
            colors['out'], fontsize=5.5, bold=True, text_color='#C62828',
            edge_color='#E53935', linewidth=1.0)

    cx5 = BX[i] + BW[i]/2
    # K-mer branch -> fusion
    arrow(cx5, ROW[0], cx5, ROW[1] + BH)
    # Assembly branch -> fusion
    arrow(cx5, ROW[2] + BH, cx5, ROW[1])
    # Fusion -> output
    arrow(cx5, ROW[1], cx5, ROW[3] - 0.15 + BH)

    # Phase 4 -> Phase 5 (features feed both branches)
    cross_arrow(BX[3] + BW[3], kmer_y + stream_h/2, BX[4], ROW[0] + BH/2)
    cross_arrow(BX[3] + BW[3], asm_y + stream_h/2, BX[4], ROW[2] + BH/2)

    # ====================================================================
    # Save
    # ====================================================================
    png_path = os.path.join(OUT_DIR, 'figure2_workflow.png')
    pdf_path = os.path.join(OUT_DIR, 'figure2_workflow.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'  Saved {png_path}')
    print(f'  Saved {pdf_path}')


if __name__ == '__main__':
    make_figure2()
