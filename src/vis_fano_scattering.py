#!/usr/bin/env python3
"""
Matplotlib visualization for Ruan & Fan (2009) TCMT Fano resonance figures.

Generates 5 publication-quality figures (12 panels total):
  1. fig1_lossless_fano.png (3 subplots): phi=0, pi/2, pi (lossless)
  2. fig2_lossy_fano.png (3 subplots): phi=0, pi/2, pi (lossy)
  3. fig4_single_channel.png (2x2): lossless C_sct, lossy C_sct, C_ext, C_abs
  4. fig5_multi_channel.png (2 subplots): total C_sct and C_abs with per-channel breakdown
  5. fig3_schematic.png (optional): MDM cylinder cross-section diagram

Dark theme: facecolor='#0d0f14', grid='#374151'

Usage:
    python src/vis_fano_scattering.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

# Dark theme colors
DARK_BG = '#0d0f14'
DARK_GRID = '#374151'
DARK_TEXT = '#e5e7eb'
ACCENT1 = '#3b82f6'
ACCENT2 = '#ef4444'
ACCENT3 = '#10b981'

def setup_dark_theme():
    """Configure matplotlib for dark theme."""
    plt.rcParams['figure.facecolor'] = DARK_BG
    plt.rcParams['axes.facecolor'] = DARK_BG
    plt.rcParams['axes.edgecolor'] = DARK_GRID
    plt.rcParams['text.color'] = DARK_TEXT
    plt.rcParams['axes.labelcolor'] = DARK_TEXT
    plt.rcParams['xtick.color'] = DARK_TEXT
    plt.rcParams['ytick.color'] = DARK_TEXT
    plt.rcParams['grid.color'] = DARK_GRID
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10


def load_csv(filepath):
    """Load CSV data, returning dict of column_name -> array."""
    data = {}
    try:
        with open(filepath, 'r') as f:
            header = f.readline().strip().split(',')
            rows = [line.strip().split(',') for line in f if line.strip()]
            for i, col in enumerate(header):
                data[col] = np.array([float(row[i]) for row in rows])
        return data
    except FileNotFoundError:
        print(f"Warning: {filepath} not found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None


def plot_fano_lineshapes(filepath, title, output_file):
    """
    Plot analytical Fano lineshapes: C_sct for phi=0, pi/2, pi.

    Creates 3-panel subplot layout.
    """
    data = load_csv(filepath)
    if data is None:
        print(f"Skipping {output_file}", file=sys.stderr)
        return

    setup_dark_theme()
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

    x = data['x']

    # Panel 1: phi = 0
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, data['c_sct_phi0'], color=ACCENT1, linewidth=2.5, label='phi=0')
    ax1.set_ylabel('C_sct', color=DARK_TEXT)
    ax1.set_title(f'{title} (phi=0)', color=DARK_TEXT, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, color=DARK_GRID)
    ax1.legend(loc='upper right', facecolor=DARK_BG, edgecolor=DARK_GRID)
    ax1.set_xlim([x.min(), x.max()])

    # Panel 2: phi = pi/2
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, data['c_sct_phi_pi2'], color=ACCENT2, linewidth=2.5, label='phi=pi/2')
    ax2.set_ylabel('C_sct', color=DARK_TEXT)
    ax2.set_title(f'{title} (phi=pi/2)', color=DARK_TEXT, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, color=DARK_GRID)
    ax2.legend(loc='upper right', facecolor=DARK_BG, edgecolor=DARK_GRID)
    ax2.set_xlim([x.min(), x.max()])

    # Panel 3: phi = pi
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x, data['c_sct_phi_pi'], color=ACCENT3, linewidth=2.5, label='phi=pi')
    ax3.set_xlabel('x = (omega - omega_0) / gamma', color=DARK_TEXT)
    ax3.set_ylabel('C_sct', color=DARK_TEXT)
    ax3.set_title(f'{title} (phi=pi)', color=DARK_TEXT, fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2, color=DARK_GRID)
    ax3.legend(loc='upper right', facecolor=DARK_BG, edgecolor=DARK_GRID)
    ax3.set_xlim([x.min(), x.max()])

    fig.suptitle(title, fontsize=14, fontweight='bold', color=DARK_TEXT, y=0.995)

    # Save with high DPI
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, facecolor=DARK_BG, dpi=100, bbox_inches='tight')
    print(f"  -> {output_file}")
    plt.close(fig)


def plot_fig4_validation(data_ll, data_lossy, output_file):
    """
    Plot Figure 4: single-channel MDM validation (2x2 grid).

    Top left: lossless C_sct (Mie vs TCMT)
    Top right: lossy C_sct (Mie vs TCMT)
    Bottom left: lossy C_ext
    Bottom right: lossy C_abs
    """
    setup_dark_theme()
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    if data_ll:
        ax1 = fig.add_subplot(gs[0, 0])
        omega_norm = data_ll['omega_norm']
        ax1.plot(omega_norm, data_ll['c_sct_mie'], 'o-', color=ACCENT1,
                 markersize=5, linewidth=2, label='Mie (numerical)', alpha=0.8)
        ax1.plot(omega_norm, data_ll['c_sct_tcmt'], '-', color=ACCENT2,
                 linewidth=2.5, label='TCMT (analytical)', alpha=0.8)
        ax1.set_ylabel('C_sct', color=DARK_TEXT)
        ax1.set_title('Lossless (gamma_d=0)', color=DARK_TEXT, fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.2, color=DARK_GRID)
        ax1.legend(loc='best', facecolor=DARK_BG, edgecolor=DARK_GRID)
        ax1.set_xlim([omega_norm.min(), omega_norm.max()])

    if data_lossy:
        # Top right: lossy C_sct
        ax2 = fig.add_subplot(gs[0, 1])
        omega_norm = data_lossy['omega_norm']
        ax2.plot(omega_norm, data_lossy['c_sct_mie'], 'o-', color=ACCENT1,
                 markersize=5, linewidth=2, label='Mie', alpha=0.8)
        ax2.plot(omega_norm, data_lossy['c_sct_tcmt'], '-', color=ACCENT2,
                 linewidth=2.5, label='TCMT', alpha=0.8)
        ax2.set_ylabel('C_sct', color=DARK_TEXT)
        ax2.set_title('Lossy (gamma_d=0.001*omega_p)', color=DARK_TEXT, fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.2, color=DARK_GRID)
        ax2.legend(loc='best', facecolor=DARK_BG, edgecolor=DARK_GRID)
        ax2.set_xlim([omega_norm.min(), omega_norm.max()])

        # Bottom left: C_ext
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(omega_norm, data_lossy['c_ext_mie'], 'o-', color=ACCENT1,
                 markersize=5, linewidth=2, label='Mie', alpha=0.8)
        ax3.plot(omega_norm, data_lossy['c_ext_tcmt'], '-', color=ACCENT2,
                 linewidth=2.5, label='TCMT', alpha=0.8)
        ax3.set_xlabel('omega / omega_p', color=DARK_TEXT)
        ax3.set_ylabel('C_ext', color=DARK_TEXT)
        ax3.set_title('Extinction cross-section', color=DARK_TEXT, fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.2, color=DARK_GRID)
        ax3.legend(loc='best', facecolor=DARK_BG, edgecolor=DARK_GRID)
        ax3.set_xlim([omega_norm.min(), omega_norm.max()])

        # Bottom right: C_abs
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(omega_norm, data_lossy['c_abs_mie'], 'o-', color=ACCENT1,
                 markersize=5, linewidth=2, label='Mie', alpha=0.8)
        ax4.plot(omega_norm, data_lossy['c_abs_tcmt'], '-', color=ACCENT2,
                 linewidth=2.5, label='TCMT', alpha=0.8)
        ax4.set_xlabel('omega / omega_p', color=DARK_TEXT)
        ax4.set_ylabel('C_abs', color=DARK_TEXT)
        ax4.set_title('Absorption cross-section', color=DARK_TEXT, fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.2, color=DARK_GRID)
        ax4.legend(loc='best', facecolor=DARK_BG, edgecolor=DARK_GRID)
        ax4.set_xlim([omega_norm.min(), omega_norm.max()])

    fig.suptitle('Figure 4: Single-Channel MDM Validation', fontsize=14, fontweight='bold', color=DARK_TEXT, y=0.995)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, facecolor=DARK_BG, dpi=100, bbox_inches='tight')
    print(f"  -> {output_file}")
    plt.close(fig)


def plot_fig5_multichannel(filepath, output_file):
    """
    Plot Figure 5: multi-channel MDM geometry.

    Left: total C_sct with per-channel breakdown
    Right: total C_abs with per-channel breakdown
    """
    data = load_csv(filepath)
    if data is None:
        print(f"Skipping {output_file}", file=sys.stderr)
        return

    setup_dark_theme()
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    omega_norm = data['omega_norm']

    # Left panel: C_sct
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(omega_norm, data['c_sct_total_mie'], 'o-', color=ACCENT1,
             markersize=6, linewidth=2.5, label='Total (Mie)', alpha=0.8)
    ax1.plot(omega_norm, data['c_sct_total_tcmt'], '-', color=ACCENT2,
             linewidth=2.5, label='Total (TCMT)', alpha=0.8)

    # Per-channel contributions (if present in CSV)
    colors = [ACCENT3, '#f59e0b', '#8b5cf6', '#ec4899']
    channels = ['c_sct_l0_mie', 'c_sct_l1_mie', 'c_sct_l2_mie']
    for i, ch in enumerate(channels):
        if ch in data:
            ax1.plot(omega_norm, data[ch], '--', color=colors[i], linewidth=1.5, alpha=0.6,
                    label=f'l={i} (Mie)')

    ax1.set_xlabel('omega / omega_p', color=DARK_TEXT)
    ax1.set_ylabel('C_sct', color=DARK_TEXT)
    ax1.set_title('Scattering cross-section', color=DARK_TEXT, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, color=DARK_GRID)
    ax1.legend(loc='best', facecolor=DARK_BG, edgecolor=DARK_GRID, fontsize=9)
    ax1.set_xlim([omega_norm.min(), omega_norm.max()])

    # Right panel: C_abs
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(omega_norm, data['c_abs_total_mie'], 'o-', color=ACCENT1,
             markersize=6, linewidth=2.5, label='Total (Mie)', alpha=0.8)
    ax2.plot(omega_norm, data['c_abs_total_tcmt'], '-', color=ACCENT2,
             linewidth=2.5, label='Total (TCMT)', alpha=0.8)

    ax2.set_xlabel('omega / omega_p', color=DARK_TEXT)
    ax2.set_ylabel('C_abs', color=DARK_TEXT)
    ax2.set_title('Absorption cross-section', color=DARK_TEXT, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, color=DARK_GRID)
    ax2.legend(loc='best', facecolor=DARK_BG, edgecolor=DARK_GRID, fontsize=9)
    ax2.set_xlim([omega_norm.min(), omega_norm.max()])

    fig.suptitle('Figure 5: Multi-Channel MDM Geometry', fontsize=14, fontweight='bold', color=DARK_TEXT, y=1.00)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, facecolor=DARK_BG, dpi=100, bbox_inches='tight')
    print(f"  -> {output_file}")
    plt.close(fig)


def main():
    """Generate all visualization figures."""
    # Define paths
    data_dir = Path('data/fano_scattering')
    img_dir = Path('data/artifacts/images')

    if not data_dir.exists():
        print(f"Data directory {data_dir} not found. Run fano-scattering binary first.", file=sys.stderr)
        sys.exit(1)

    print("Generating Fano TCMT visualization figures...")

    # Figure 1: Lossless analytical Fano
    print("Generating Figure 1...")
    plot_fano_lineshapes(
        data_dir / 'fig1_lossless_fano.csv',
        'Figure 1: Lossless Analytical Fano',
        img_dir / 'fano_fig1_lossless.png'
    )

    # Figure 2: Lossy analytical Fano
    print("Generating Figure 2...")
    plot_fano_lineshapes(
        data_dir / 'fig2_lossy_fano.csv',
        'Figure 2: Lossy Analytical Fano',
        img_dir / 'fano_fig2_lossy.png'
    )

    # Figure 4: Single-channel MDM validation
    print("Generating Figure 4...")
    data_ll = load_csv(data_dir / 'fig4a_lossless.csv')
    data_lossy = load_csv(data_dir / 'fig4bcd_lossy.csv')
    plot_fig4_validation(data_ll, data_lossy, img_dir / 'fano_fig4_validation.png')

    # Figure 5: Multi-channel MDM
    print("Generating Figure 5...")
    plot_fig5_multichannel(
        data_dir / 'fig5_multi.csv',
        img_dir / 'fano_fig5_multichannel.png'
    )

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
