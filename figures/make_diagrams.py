"""Render the concept diagrams used in the lesson notebooks (01: figures 01-07, 02: figures 08-11).

Run from the workspace root:  conda run -n fo_lerobot python figures/make_diagrams.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "none",
})

# palette
INK = "#1f2933"
MUTED = "#7b8794"
LINE = "#cbd2d9"
IMG = "#3e7cb1"      # camera / image
STATE = "#2f9e78"    # observation.state
ACTION = "#e07a1f"   # action
BOOK = "#9aa5b1"     # bookkeeping
PAD = "#d64545"      # padding
BATCH = "#7c5cbf"    # batch axis
LEFT = "#2f9e78"
RIGHT = "#3e7cb1"
GRIP = "#e07a1f"


def box(ax, x, y, w, h, color, text="", fc=None, lw=1.4, fontsize=10, textcolor=None, style="round,pad=0.02,rounding_size=0.06", alpha=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, ec=color, fc=fc or color, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=textcolor or INK, zorder=z + 1)
    return p


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6, style="-|>", ms=14, z=3, ls="-", conn="arc3,rad=0"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z, linestyle=ls, connectionstyle=conn)
    ax.add_patch(a)


def label(ax, x, y, s, size=10, color=INK, ha="center", va="center", weight="normal", family=None, z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, weight=weight, family=family, zorder=z)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


# ----------------------------------------------------------------------------------------------
# 1. the flat table
# ----------------------------------------------------------------------------------------------
def dataset_table():
    fig, ax = canvas(12.6, 5.6)
    cols = [
        ("observation.images.top", "video → mp4", IMG, 2.6),
        ("observation.state", "float32 (14,)", STATE, 1.7),
        ("action", "float32 (14,)", ACTION, 1.7),
        ("episode_index", "int64", BOOK, 1.5),
        ("frame_index", "int64", BOOK, 1.35),
        ("timestamp", "float32", BOOK, 1.15),
    ]
    x0, top = 1.35, 4.35
    rh = 0.34
    rows = [
        ("row 0", ["frame @ 0.00 s", "[0.01, −0.19, …]", "[0.01, −0.18, …]", "0", "0", "0.00"], 0),
        ("row 1", ["frame @ 0.02 s", "[0.01, −0.19, …]", "[0.01, −0.18, …]", "0", "1", "0.02"], 0),
        ("⋮", ["⋮"] * 6, 0),
        ("row 499", ["frame @ 9.98 s", "[0.02, −0.10, …]", "[0.02, −0.11, …]", "0", "499", "9.98"], 0),
        ("row 500", ["frame @ 0.00 s", "[0.00, −0.21, …]", "[0.00, −0.20, …]", "1", "0", "0.00"], 1),
        ("⋮", ["⋮"] * 6, 1),
        ("row 999", ["frame @ 9.98 s", "[0.03, −0.12, …]", "[0.03, −0.12, …]", "1", "499", "9.98"], 1),
        ("⋮", ["⋮"] * 6, 2),
        ("row 24999", ["frame @ 9.98 s", "[0.01, −0.09, …]", "[0.01, −0.10, …]", "49", "499", "9.98"], 49),
    ]
    # header
    x = x0
    for name, dtype, color, w in cols:
        box(ax, x, top, w - 0.06, 0.62, color, fc=color, alpha=0.92)
        label(ax, x + (w - 0.06) / 2, top + 0.40, name, size=8.8, color="white", weight="bold", family="DejaVu Sans Mono")
        label(ax, x + (w - 0.06) / 2, top + 0.17, dtype, size=8.5, color="white")
        x += w
    table_w = x - x0
    # rows
    y = top
    shade = {0: "#eef6fb", 1: "#f3f8f3", 2: "#f7f7f7", 49: "#fbf4ee"}
    for rname, vals, ep in rows:
        y -= rh
        ax.add_patch(Rectangle((x0, y), table_w - 0.06, rh, fc=shade[ep], ec="none", zorder=1))
        label(ax, x0 - 0.12, y + rh / 2, rname, size=9, color=MUTED, ha="right", family=None if rname == "⋮" else "DejaVu Sans Mono")
        x = x0
        for (name, dtype, color, w), v in zip(cols, vals):
            label(ax, x + (w - 0.06) / 2, y + rh / 2, v, size=8.8, color=INK if v != "⋮" else MUTED, family=None if v == "⋮" else "DejaVu Sans Mono")
            x += w
    # episode braces
    def brace(y_top, y_bot, text, color):
        bx = x0 + table_w + 0.12
        ax.plot([bx, bx + 0.12, bx + 0.12, bx], [y_top, y_top, y_bot, y_bot], color=color, lw=1.6)
        label(ax, bx + 0.22, (y_top + y_bot) / 2, text, size=9.5, color=color, ha="left")
    brace(top, top - 4 * rh, "episode 0\n500 rows", STATE)
    brace(top - 4 * rh, top - 7 * rh, "episode 1\n500 rows", IMG)
    brace(top - 8 * rh, top - 9 * rh, "episode 49", ACTION)
    # mp4 note
    label(ax, x0, top - 9 * rh - 0.45, "pixels live in mp4 files; the table stores which video and which timestamp each row is",
          size=8.8, color=IMG, ha="left")
    label(ax, 6.3, 5.4, "One flat table: one row per frame, one column per feature. Episodes are contiguous row ranges.",
          size=11.5, color=INK, weight="bold")
    label(ax, 6.3, 0.22, "25 000 rows  =  50 episodes  ×  500 frames", size=10, color=MUTED)
    save(fig, "01_dataset_table.png")


# ----------------------------------------------------------------------------------------------
# 2. one frame
# ----------------------------------------------------------------------------------------------
def cube(ax, x, y, w, h, depth, color, n=3):
    for i in reversed(range(n)):
        dx = i * depth
        ax.add_patch(Rectangle((x + dx, y + dx), w, h, fc=color, ec="white", lw=1.2, alpha=0.55 + 0.15 * (n - i), zorder=2 + (n - i)))


def one_frame():
    fig, ax = canvas(12, 6.4)
    label(ax, 6, 6.1, "ds[0]  →  one row of the table, as a dict of tensors", size=12.5, weight="bold")

    # image
    label(ax, 0.4, 5.05, '"observation.images.top"', size=10, ha="left", family="DejaVu Sans Mono", color=IMG, weight="bold")
    cube(ax, 0.55, 2.55, 2.2, 1.65, 0.14, IMG)
    label(ax, 1.65 + 0.14, 2.3, "W = 640", size=9, color=MUTED)
    ax.text(0.42, 3.4, "H = 480", rotation=90, ha="center", va="center", fontsize=9, color=MUTED)
    label(ax, 3.2, 4.25, "C = 3", size=9, color=MUTED)
    label(ax, 1.75, 1.85, "shape (3, 480, 640)\nfloat32 in [0, 1]\ndecoded from the mp4", size=9.5, color=INK)

    # state vector
    def vector(y, title, color, note):
        label(ax, 4.6, y + 0.62, title, size=10, ha="left", family="DejaVu Sans Mono", color=color, weight="bold")
        cw = 0.46
        for i in range(14):
            c = GRIP if i in (6, 13) else (LEFT if i < 6 else RIGHT)
            ax.add_patch(Rectangle((4.6 + i * cw, y), cw - 0.04, 0.42, fc=c, ec="none", alpha=0.85, zorder=2))
            label(ax, 4.6 + i * cw + (cw - 0.04) / 2, y + 0.21, str(i), size=8.5, color="white", weight="bold")
        label(ax, 4.6 + 14 * cw + 0.15, y + 0.21, note, size=9.5, ha="left", color=INK)
        # group labels
        ax.plot([4.62, 4.6 + 6 * cw - 0.06], [y - 0.09, y - 0.09], color=LEFT, lw=1.4)
        label(ax, 4.6 + 3 * cw, y - 0.26, "left arm, 6 joints", size=8.5, color=LEFT)
        label(ax, 4.6 + 6 * cw + (cw - 0.04) / 2, y - 0.26, "gripper", size=8.5, color=GRIP)
        ax.plot([4.6 + 7 * cw + 0.02, 4.6 + 13 * cw - 0.06], [y - 0.09, y - 0.09], color=RIGHT, lw=1.4)
        label(ax, 4.6 + 10 * cw, y - 0.26, "right arm, 6 joints", size=8.5, color=RIGHT)
        label(ax, 4.6 + 13 * cw + (cw - 0.04) / 2, y - 0.26, "gripper", size=8.5, color=GRIP)

    vector(4.2, '"observation.state"', STATE, "shape (14,)\nmeasured joint positions")
    vector(2.6, '"action"', ACTION, "shape (14,)\ncommanded joint positions")

    # scalars + task
    y = 1.35
    label(ax, 4.6, y + 0.52, "bookkeeping, shape ()", size=9.5, ha="left", color=MUTED)
    for i, (k, v) in enumerate([("episode_index", "0"), ("frame_index", "0"), ("timestamp", "0.0"), ("index", "0"), ("task_index", "0"), ("next.done", "False")]):
        bx = 4.6 + i * 1.22
        box(ax, bx, y - 0.25, 1.14, 0.5, BOOK, fc="#f1f3f5", lw=1.0)
        label(ax, bx + 0.57, y + 0.08, k, size=7.4, family="DejaVu Sans Mono", color=INK)
        label(ax, bx + 0.57, y - 0.12, v, size=8.5, color=MUTED, family="DejaVu Sans Mono")
    label(ax, 4.6, 0.45, '"task"', size=10, ha="left", family="DejaVu Sans Mono", color=INK, weight="bold")
    box(ax, 5.45, 0.22, 4.4, 0.46, LINE, fc="#fafafa", lw=1.0)
    label(ax, 7.65, 0.45, "Insert the peg into the socket.", size=9.5, color=INK, family="DejaVu Sans Mono")
    label(ax, 10.05, 0.45, "str, from tasks.parquet", size=9, ha="left", color=MUTED)
    label(ax, 6, -0.1, "State and action share the same 14 slots: 0–5 left arm, 6 left gripper, 7–12 right arm, 13 right gripper.", size=9.5, color=MUTED)
    ax.set_ylim(-0.3, 6.4)
    save(fig, "02_one_frame.png")


# ----------------------------------------------------------------------------------------------
# 3. episode matrix, real data
# ----------------------------------------------------------------------------------------------
def episode_matrix(actions, fps):
    T, D = actions.shape
    t = np.arange(T) / fps
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.42)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    im = ax0.imshow(actions, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2, extent=[-0.5, D - 0.5, t[-1], 0])
    ax0.set_title("actions of episode 0:  matrix (500, 14)", fontsize=11, color=INK, loc="left", pad=16)
    ax0.set_xlabel("joint (column)")
    ax0.set_ylabel("time (row), seconds")
    ax0.set_xticks(range(D))
    ax0.set_xticklabels([str(i) for i in range(D)], fontsize=8)
    for sp in ax0.spines.values():
        sp.set_visible(False)
    for i in range(D):
        c = GRIP if i in (6, 13) else (LEFT if i < 6 else RIGHT)
        ax0.add_patch(Rectangle((i - 0.5, -0.35), 1, 0.3, fc=c, ec="none", clip_on=False))
    cb = fig.colorbar(im, ax=ax0, orientation="horizontal", fraction=0.05, pad=0.12)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    for i in range(D):
        c = GRIP if i in (6, 13) else (LEFT if i < 6 else RIGHT)
        ax1.plot(t, actions[:, i], color=c, lw=1.4, alpha=0.9 if i in (6, 13) else 0.7)
    ax1.set_title("each column is one line:  action[:, d]  against time", fontsize=11, color=INK, loc="left")
    ax1.set_xlabel("seconds")
    ax1.set_ylabel("commanded position")
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["left"].set_color(LINE)
    ax1.spines["bottom"].set_color(LINE)
    ax1.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[Line2D([], [], color=LEFT, lw=2, label="left arm joints (0–5)"),
                        Line2D([], [], color=RIGHT, lw=2, label="right arm joints (7–12)"),
                        Line2D([], [], color=GRIP, lw=2, label="grippers (6, 13)")],
               frameon=False, fontsize=9, loc="upper right")
    # connector arrow between panels
    fig.text(0.415, 0.56, "→", fontsize=26, color=MUTED, ha="center", va="center")
    fig.text(0.415, 0.49, "one line\nper column", fontsize=8.5, color=MUTED, ha="center", va="center")
    save(fig, "03_episode_matrix.png")


# ----------------------------------------------------------------------------------------------
# 4. delta_timestamps
# ----------------------------------------------------------------------------------------------
def delta_timestamps():
    fig, ax = canvas(13, 6.2)
    label(ax, 6.5, 5.9, "delta_timestamps: pick rows relative to frame t, stack them on a new leading time axis", size=12, weight="bold")

    # timeline of rows
    cols = ["t−2", "t−1", "t", "t+1", "t+2", "t+3", "…", "t+9", "t+10"]
    cw, y_row = 0.68, 4.75
    x0 = 2.75
    label(ax, x0 - 0.15, y_row + 0.22, "episode rows", size=9.5, ha="right", color=MUTED)
    for i, c in enumerate(cols):
        x = x0 + i * cw
        is_t = c == "t"
        ax.add_patch(Rectangle((x, y_row), cw - 0.06, 0.44, fc="#f1f3f5" if not is_t else INK, ec=LINE if not is_t else INK, lw=1, zorder=2))
        label(ax, x + (cw - 0.06) / 2, y_row + 0.22, c, size=9.5, color="white" if is_t else INK, family="DejaVu Sans Mono")
    # arrows to the right = time
    arrow(ax, x0 + 9 * cw + 0.05, y_row + 0.22, x0 + 9 * cw + 0.6, y_row + 0.22, color=MUTED)
    label(ax, x0 + 9 * cw + 0.75, y_row + 0.22, "time", size=9, ha="left", color=MUTED)

    def pick(y, title, color, idxs, offsets_text, shape_before, shape_after, n):
        label(ax, 0.25, y + 0.2, title, size=9.5, ha="left", family="DejaVu Sans Mono", color=color, weight="bold")
        label(ax, 0.25, y - 0.12, offsets_text, size=8.6, ha="left", color=MUTED, family="DejaVu Sans Mono")
        # highlight picked rows on timeline by dropping brackets
        for j, i in enumerate(idxs):
            x = x0 + i * cw + (cw - 0.06) / 2
            ax.add_patch(Rectangle((x0 + i * cw, y - 0.02), cw - 0.06, 0.44, fc=color, ec="none", alpha=0.85, zorder=2))
            label(ax, x, y + 0.2, cols[i], size=8.8, color="white", family="DejaVu Sans Mono")
            arrow(ax, x, y_row - 0.02, x, y + 0.46, color=color, lw=1.0, ms=9, ls=(0, (3, 3)))
        # stack arrow + result
        xr = x0 + 9 * cw + 0.55
        arrow(ax, xr - 0.35, y + 0.2, xr + 0.25, y + 0.2, color=color)
        label(ax, xr + 0.35, y + 0.2, "stack", size=8.5, ha="left", color=MUTED)
        # stacked slabs
        sx = xr + 1.05
        for k in range(min(n, 4)):
            ax.add_patch(Rectangle((sx + k * 0.09, y - 0.04 + k * 0.07), 0.9, 0.4, fc=color, ec="white", lw=1, alpha=0.75, zorder=3 + k))
        if n > 4:
            label(ax, sx + 0.45 + 0.3, y + 0.55, "…", size=12, color=color)
        label(ax, sx + 1.55, y + 0.3, shape_after, size=9.5, ha="left", family="DejaVu Sans Mono", color=INK, weight="bold")
        label(ax, sx + 1.55, y + 0.02, f"was {shape_before}", size=8.5, ha="left", family="DejaVu Sans Mono", color=MUTED)

    pick(3.55, '"observation.images.top"', IMG, [1, 2], "offsets [-1/fps, 0]", "(3, 480, 640)", "(2, 3, 480, 640)", 2)
    pick(2.45, '"observation.state"', STATE, [1, 2], "offsets [-1/fps, 0]", "(14,)", "(2, 14)", 2)
    pick(1.35, '"action"', ACTION, [2, 3, 4, 5, 6, 7], "offsets [0, 1/fps, …, 9/fps]", "(14,)", "(10, 14)", 10)

    label(ax, 6.5, 0.45, "The new first axis is time; its length is len(offsets). History for observations, a chunk of future commands for action.",
          size=9.8, color=MUTED)
    save(fig, "04_delta_timestamps.png")


# ----------------------------------------------------------------------------------------------
# 5. padding at the episode end
# ----------------------------------------------------------------------------------------------
def padding():
    fig, ax = canvas(12, 4.2)
    label(ax, 6, 3.9, "At the last frame (row 499) the action offsets run past the episode: rows are clamped and flagged", size=12, weight="bold")
    wanted = [499 + i for i in range(10)]
    cw, x0 = 0.95, 1.25
    y_w, y_r, y_p = 2.7, 1.75, 0.9
    label(ax, x0 - 0.15, y_w + 0.22, "wanted rows", size=9.5, ha="right", color=MUTED)
    label(ax, x0 - 0.15, y_r + 0.22, "rows returned", size=9.5, ha="right", color=MUTED)
    label(ax, x0 - 0.15, y_p + 0.22, "action_is_pad", size=9.5, ha="right", color=MUTED, family="DejaVu Sans Mono")
    for i, r in enumerate(wanted):
        x = x0 + i * cw
        inside = r <= 499
        ax.add_patch(Rectangle((x, y_w), cw - 0.08, 0.44, fc="#f1f3f5" if inside else "#fbe9e9", ec=LINE if inside else PAD, lw=1, ls="-" if inside else "--", zorder=2))
        label(ax, x + (cw - 0.08) / 2, y_w + 0.22, str(r), size=9.5, color=INK if inside else PAD, family="DejaVu Sans Mono")
        ax.add_patch(Rectangle((x, y_r), cw - 0.08, 0.44, fc=ACTION, ec="none", alpha=0.85, zorder=2))
        label(ax, x + (cw - 0.08) / 2, y_r + 0.22, "499", size=9.5, color="white", family="DejaVu Sans Mono")
        arrow(ax, x + (cw - 0.08) / 2, y_w - 0.02, x + (cw - 0.08) / 2, y_r + 0.48, color=MUTED if inside else PAD, lw=1.0, ms=9)
        flag = "False" if inside else "True"
        ax.add_patch(Rectangle((x, y_p), cw - 0.08, 0.44, fc="#e6f4ee" if inside else PAD, ec="none", zorder=2))
        label(ax, x + (cw - 0.08) / 2, y_p + 0.22, flag, size=9.2, color=STATE if inside else "white", family="DejaVu Sans Mono", weight="bold")
    # episode end line
    xe = x0 + cw - 0.04
    ax.plot([xe, xe], [y_p - 0.15, y_w + 0.6], color=PAD, lw=1.6, ls="--")
    label(ax, xe + 0.06, y_w + 0.72, "episode ends after row 499", size=9, ha="left", color=PAD)
    label(ax, 6, 0.3, "action still has shape (10, 14); the mask tells the loss which of the 10 rows are real. Same rule for every key with offsets.", size=9.6, color=MUTED)
    save(fig, "05_padding.png")


# ----------------------------------------------------------------------------------------------
# 6. batch
# ----------------------------------------------------------------------------------------------
def batch():
    fig, ax = canvas(13.4, 6.0)
    label(ax, 6.7, 5.7, "DataLoader(batch_size=8): stack 8 frames on a new leading batch axis", size=12, weight="bold")
    # 8 frame cards
    card_w, card_h = 1.05, 3.0
    x0, y0 = 0.5, 1.6
    keys = [("images", "(2,3,480,640)", IMG), ("state", "(2,14)", STATE), ("action", "(10,14)", ACTION), ("is_pad", "(10,)", PAD), ("task", "str", BOOK)]
    for k in range(8):
        x = x0 + k * (card_w + 0.08)
        box(ax, x, y0, card_w, card_h, LINE, fc="#fafafa", lw=1.0)
        label(ax, x + card_w / 2, y0 + card_h - 0.2, f"ds[i{k + 1}]", size=8.5, color=MUTED, family="DejaVu Sans Mono")
        for j, (name, shp, c) in enumerate(keys):
            yy = y0 + card_h - 0.6 - j * 0.5
            ax.add_patch(Rectangle((x + 0.08, yy - 0.16), card_w - 0.16, 0.36, fc=c, ec="none", alpha=0.8, zorder=2))
            label(ax, x + card_w / 2, yy + 0.06, name, size=7.8, color="white", weight="bold")
            label(ax, x + card_w / 2, yy - 0.09, shp, size=6.6, color="white", family="DejaVu Sans Mono")
    xr = x0 + 8 * (card_w + 0.08) + 0.15
    arrow(ax, xr - 0.05, y0 + card_h / 2, xr + 0.55, y0 + card_h / 2, color=BATCH, lw=2.2, ms=18)
    label(ax, xr + 0.25, y0 + card_h / 2 + 0.3, "collate", size=9, color=BATCH)
    # batch result
    bx = xr + 0.75
    box(ax, bx, y0 - 0.05, 2.15, card_h + 0.1, BATCH, fc="#f4f0fb", lw=1.4)
    label(ax, bx + 1.075, y0 + card_h - 0.2, "batch", size=9.5, color=BATCH, weight="bold")
    out = [("images", "(8, 2, 3, 480, 640)", IMG), ("state", "(8, 2, 14)", STATE), ("action", "(8, 10, 14)", ACTION), ("action_is_pad", "(8, 10)", PAD), ("task", "list of 8 str", BOOK)]
    for j, (name, shp, c) in enumerate(out):
        yy = y0 + card_h - 0.6 - j * 0.5
        ax.add_patch(Rectangle((bx + 0.1, yy - 0.16), 1.95, 0.36, fc=c, ec="none", alpha=0.85, zorder=2))
        label(ax, bx + 1.075, yy + 0.06, name, size=8, color="white", weight="bold")
        label(ax, bx + 1.075, yy - 0.09, shp, size=7.2, color="white", family="DejaVu Sans Mono")
    # axis legend
    label(ax, 6.7, 0.95, "read every batch shape as  ( batch ,  time ,  feature… )", size=11, color=INK)
    label(ax, 6.7, 0.5, "action (8, 10, 14)  =  8 frames  ×  10 future timesteps  ×  14 joints", size=10, color=MUTED, family="DejaVu Sans Mono")
    for txt, c, xx in [("batch", BATCH, 4.25), ("time", ACTION, 5.7), ("feature", STATE, 7.15)]:
        pass
    save(fig, "06_batch.png")


# ----------------------------------------------------------------------------------------------
# 7. shape flow (recap)
# ----------------------------------------------------------------------------------------------
def shape_flow():
    fig, ax = canvas(12, 4.6)
    stages = ["on disk", "one row  ds[i]", "+ delta_timestamps", "+ DataLoader(batch_size=8)"]
    axes_txt = ["", "(feature…)", "(time, feature…)", "(batch, time, feature…)"]
    xs = [0.3, 3.05, 5.9, 8.9]
    ws = [2.35, 2.45, 2.6, 2.85]
    rows = [
        ("mp4 video", "image  (3, 480, 640)", "(2, 3, 480, 640)", "(8, 2, 3, 480, 640)", IMG),
        ("parquet column", "state  (14,)", "(2, 14)", "(8, 2, 14)", STATE),
        ("parquet column", "action (14,)", "(10, 14)", "(8, 10, 14)", ACTION),
        ("", "", "action_is_pad (10,)", "(8, 10)", PAD),
        ("tasks.parquet", "task   str", "str", "list of 8 str", BOOK),
    ]
    top = 3.75
    for x, w, s, a in zip(xs, ws, stages, axes_txt):
        label(ax, x + w / 2, top + 0.45, s, size=10.5, weight="bold", color=INK)
        label(ax, x + w / 2, top + 0.17, a, size=8.8, color=MUTED, family="DejaVu Sans Mono")
        ax.plot([x, x + w], [top, top], color=LINE, lw=1.2)
    for j, (a, b, c, d, color) in enumerate(rows):
        y = top - 0.55 - j * 0.6
        for x, w, txt in zip(xs, ws, [a, b, c, d]):
            if txt:
                ax.add_patch(Rectangle((x, y - 0.2), w, 0.42, fc=color, ec="none", alpha=0.12 if txt else 0, zorder=1))
                label(ax, x + w / 2, y, txt, size=9.2, family="DejaVu Sans Mono", color=INK)
        for k in range(3):
            if [a, b, c, d][k] and [a, b, c, d][k + 1]:
                arrow(ax, xs[k] + ws[k] + 0.03, y, xs[k + 1] - 0.03, y, color=color, lw=1.4, ms=11)
    label(ax, 6, 0.25, "Every arrow is one thing you did in this notebook.", size=9.8, color=MUTED)
    save(fig, "07_shape_flow.png")


# ==============================================================================================
# Lesson 2
# ==============================================================================================
FO = "#ff6d04"       # FiftyOne orange
FO_DARK = "#c2530a"
FOLDER = "#8a6d3b"


def folder(ax, x, y, w, h, title, color=FOLDER, fc="#fbf7ef"):
    """A folder-shaped box with a tab."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", ec=color, fc=fc, lw=1.4, zorder=2))
    ax.add_patch(Rectangle((x + 0.05, y + h - 0.02), w * 0.42, 0.22, fc=color, ec="none", zorder=2))
    label(ax, x + 0.05 + w * 0.21, y + h + 0.09, title, size=8.5, color="white", weight="bold", family="DejaVu Sans Mono")


def sample_card(ax, x, y, w, h, title="sample", fields=None, ref_lines=None, fontsize=8.6):
    box(ax, x, y, w, h, FO, fc="#fff7f0", lw=1.6)
    label(ax, x + w / 2, y + h - 0.22, title, size=10, color=FO_DARK, weight="bold")
    yy = y + h - 0.55
    for k, v in (fields or []):
        label(ax, x + 0.15, yy, k, size=fontsize, ha="left", family="DejaVu Sans Mono", color=INK)
        label(ax, x + w - 0.15, yy, v, size=fontsize, ha="right", family="DejaVu Sans Mono", color=MUTED)
        yy -= 0.3
    if ref_lines:
        yy -= 0.05
        ax.plot([x + 0.12, x + w - 0.12], [yy + 0.16, yy + 0.16], color=LINE, lw=1)
        for line, c in ref_lines:
            label(ax, x + 0.15, yy, line, size=fontsize, ha="left", family="DejaVu Sans Mono", color=c)
            yy -= 0.3
    return yy


# ----------------------------------------------------------------------------------------------
# 8. one FiftyOne sample = one episode, pointing into the LeRobot source
# ----------------------------------------------------------------------------------------------
def episode_sample():
    fig, ax = canvas(13, 6.6)
    label(ax, 6.5, 6.3, "A FiftyOne sample is one episode: a handful of scalars plus a pointer into the LeRobot source", size=12, weight="bold")

    # left: FiftyOne dataset of 50 samples, one expanded
    label(ax, 2.35, 5.7, "FiftyOne dataset  \u00b7  50 samples", size=10, color=FO_DARK, weight="bold")
    cy, ch = 2.0, 3.3
    for k in range(6, 0, -1):
        ax.add_patch(Rectangle((0.55 + k * 0.07, cy - k * 0.07), 3.6, ch, fc="#fff7f0", ec=FO, lw=0.8, alpha=0.5, zorder=1))
    sample_card(ax, 0.55, cy, 3.6, ch, title="sample  (episode 0)",
                fields=[("episode_index", "0"), ("task", '"Insert the peg\u2026"'), ("length", "500"), ("duration", "10.0"), ("fps", "50.0"), ("robot_type", '"aloha"')],
                ref_lines=[("media_reference", FO_DARK),
                           ("  .data   = [0, 0, 0, 500]", IMG),
                           ("  .videos = {top: [0, 0, 0.0, 10.0]}", ACTION)])
    label(ax, 2.35, 1.25, "filterable scalars above the line;\nthe pointer below it", size=8.8, color=MUTED)

    # right: LeRobot folder with data table and video
    fx, fy, fw, fh = 5.2, 0.7, 7.3, 4.75
    ax.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0.02,rounding_size=0.08", ec=FOLDER, fc="#fbf7ef", lw=1.4, zorder=2))
    ax.add_patch(Rectangle((fx + 0.05, fy + fh - 0.02), fw * 0.72, 0.24, fc=FOLDER, ec="none", zorder=2))
    label(ax, fx + 0.15, fy + fh + 0.1, "~/lerobot-data/aloha_sim_insertion_human/", size=8.5, color="white", weight="bold", family="DejaVu Sans Mono", ha="left")
    label(ax, fx + fw / 2, fy + fh + 0.42, 'dataset.media_sources[0]["loc"]  \u2192  the LeRobot folder on disk  (stored once per dataset)', size=9, color=FOLDER, family="DejaVu Sans Mono")

    # data table
    label(ax, 5.55, 4.95, "data/\u2026/file-000.parquet   (25 000 rows)", size=9, ha="left", family="DejaVu Sans Mono", color=INK)
    tx, ty, tw = 5.55, 1.55, 3.0
    n = 12
    rh = 0.26
    names = ["row 0", "row 1", "\u2026", "row 499", "row 500", "\u2026", "row 999", "\u2026", "\u2026", "\u2026", "\u2026", "row 24999"]
    for i in range(n):
        y = ty + (n - 1 - i) * rh
        in_ep0 = i < 4
        ax.add_patch(Rectangle((tx, y), tw, rh - 0.04, fc=IMG if in_ep0 else "#eceff1", ec="none", alpha=0.85 if in_ep0 else 1, zorder=2))
        txt = names[i]
        label(ax, tx + 0.12, y + rh / 2 - 0.02, txt, size=7.8, ha="left", family=None if txt == "\u2026" else "DejaVu Sans Mono", color="white" if in_ep0 else MUTED)
    by0, by1 = ty + (n - 4) * rh, ty + n * rh - 0.04
    ax.plot([tx + tw + 0.1, tx + tw + 0.22, tx + tw + 0.22, tx + tw + 0.1], [by1, by1, by0, by0], color=IMG, lw=1.4)
    label(ax, tx + tw + 0.32, (by0 + by1) / 2, "rows 0:500\n= .data[2:4]", size=8.5, ha="left", color=IMG)

    # video
    vx, vy, vw = 9.7, 2.55, 2.55
    label(ax, vx, 3.55, "videos/observation.images.top/\n\u2026/file-000.mp4", size=8.4, ha="left", family="DejaVu Sans Mono", color=INK)
    ax.add_patch(Rectangle((vx, vy), vw, 0.5, fc="#eceff1", ec="none", zorder=2))
    ax.add_patch(Rectangle((vx, vy), vw * 0.1, 0.5, fc=ACTION, ec="none", alpha=0.9, zorder=3))
    for k in range(1, 10):
        ax.plot([vx + vw * k / 10] * 2, [vy, vy + 0.5], color="white", lw=1, zorder=4)
    label(ax, vx, vy - 0.17, "0 s", size=7.5, color=MUTED)
    label(ax, vx + vw, vy - 0.17, "500 s", size=7.5, color=MUTED)
    label(ax, vx + vw * 0.1 + 0.1, vy + 0.25, "0.0 \u2013 10.0 s = .videos[top][2:4]", size=8.0, color=ACTION, ha="left")
    label(ax, vx + vw / 2, vy - 0.6, "50 episodes share one mp4;\nthe pointer is a time window", size=8.3, color=MUTED)

    # arrows from the two pointer lines to the folder pieces
    arrow(ax, 4.2, 2.6, tx - 0.05, (by0 + by1) / 2, color=IMG, lw=1.6)
    arrow(ax, 4.2, 2.3, vx - 0.05, vy + 0.25, color=ACTION, lw=1.6)
    label(ax, 6.5, 0.28, "Nothing frame-level lives in FiftyOne's database. The App, and your code, read frames through the pointer.", size=9.5, color=MUTED)
    save(fig, "08_episode_sample.png")


# ----------------------------------------------------------------------------------------------
# 9. road 1: following the pointer to a (500, 14) array
# ----------------------------------------------------------------------------------------------
def pointer_road():
    fig, ax = canvas(13, 4.6)
    label(ax, 6.5, 4.3, "Road 1, read in place: two values from FiftyOne, one slice in LeRobot", size=12, weight="bold")

    # step boxes
    def step(x, y, w, h, title, code, color, sub=""):
        box(ax, x, y, w, h, color, fc="white", lw=1.6)
        label(ax, x + w / 2, y + h - 0.28, title, size=9.5, color=color, weight="bold")
        label(ax, x + w / 2, y + h / 2 - 0.02, code, size=8.6, family="DejaVu Sans Mono", color=INK)
        if sub:
            label(ax, x + w / 2, y + 0.25, sub, size=8.3, color=MUTED)

    step(0.3, 2.35, 3.7, 1.5, "where the files are", 'dataset.media_sources[0]["loc"]', FOLDER, "once per dataset → a folder path")
    step(0.3, 0.45, 3.7, 1.5, "which rows are this episode", "sample.media_reference.data", IMG, "once per sample → [chunk, file, 0, 500]")

    step(4.75, 1.4, 3.6, 1.5, "open the folder (Lesson 1)", "LeRobotDataset(repo_id, root=root)", INK, "the native torch Dataset, no download")
    arrow(ax, 4.0, 3.1, 4.75, 2.3, color=FOLDER)
    step(8.9, 1.4, 3.85, 1.5, "slice the frame table", 'lr.hf_dataset["action"][row_start:row_end]', ACTION, "→ actions of episode 0, shape (500, 14)")
    arrow(ax, 8.35, 2.15, 8.9, 2.15, color=INK)
    arrow(ax, 4.0, 0.9, 10.8, 1.38, color=IMG, ls=(0, (4, 3)), conn="arc3,rad=0.25")
    label(ax, 7.0, 0.62, "row_start, row_end", size=8, color=IMG, family="DejaVu Sans Mono")

    label(ax, 6.5, 0.15, "Same array as Lesson 1 section 5, reached from a FiftyOne sample instead of a hard-coded index. No data copied.", size=9.5, color=MUTED)
    save(fig, "09_pointer_road.png")


# ----------------------------------------------------------------------------------------------
# 10. three roads from a view back to LeRobot
# ----------------------------------------------------------------------------------------------
def three_roads():
    fig, ax = canvas(13, 7.2)
    label(ax, 6.5, 6.9, "Three roads from a FiftyOne view back to frames. Pick by asking: who consumes the frames?", size=12, weight="bold")

    # the view
    box(ax, 0.4, 2.5, 2.6, 2.2, FO, fc="#fff7f0", lw=1.8)
    label(ax, 1.7, 4.35, "FiftyOne view", size=10.5, color=FO_DARK, weight="bold")
    label(ax, 1.7, 3.85, "dataset.take(10)\ndataset.match(...)\nsort_by_similarity(...)", size=8.6, family="DejaVu Sans Mono", color=INK)
    label(ax, 1.7, 2.95, "a query, nothing copied\n→ a set of episode samples", size=8.5, color=MUTED)

    def road(y, num, title, consumer, code, copies, color, use_when):
        x = 4.1
        arrow(ax, 3.0, 3.6, x, y + 0.75, color=color, lw=2.0, ms=16)
        box(ax, x, y, 8.6, 1.55, color, fc="white", lw=1.6)
        ax.add_patch(Rectangle((x, y), 0.55, 1.55, fc=color, ec="none", zorder=3))
        label(ax, x + 0.275, y + 0.775, str(num), size=16, color="white", weight="bold")
        label(ax, x + 0.75, y + 1.25, title, size=10.5, color=color, weight="bold", ha="left")
        label(ax, x + 0.75, y + 0.9, code, size=8.6, family="DejaVu Sans Mono", color=INK, ha="left")
        label(ax, x + 0.75, y + 0.55, "consumer:  " + consumer, size=8.8, color=INK, ha="left")
        label(ax, x + 0.75, y + 0.22, use_when, size=8.3, color=MUTED, ha="left")
        # copies badge
        bc = PAD if copies else STATE
        box(ax, x + 6.95, y + 1.05, 1.5, 0.38, bc, fc=bc, lw=0)
        label(ax, x + 7.7, y + 1.24, "copies data" if copies else "no copy", size=8.5, color="white", weight="bold")

    road(4.9, 1, "Read in place", "your own code, one sample at a time",
         'root = media_sources[0]["loc"];  rows = media_reference.data[2:4]',
         False, IMG, "you hold a sample and need its rows now: a plot, a metric, a policy on one episode")
    road(2.75, 2, "Torch dataset", "a training or inference loop",
         'view.to_torch(GetItem)   or   LeRobotDataset(root, episodes=view.values(...))',
         False, BATCH, "batches from a curated subset; per episode or per frame")
    road(0.6, 3, "Export", "anything outside this Python process",
         "view.export(export_dir, dataset_type=fo.types.LeRobotDataset)",
         True, ACTION, "a new standalone LeRobot v3 dataset: training on another machine, Hub upload, a colleague")

    label(ax, 6.5, 0.2, "Rule of thumb:  in-process → do not copy (1, 2).   Out-of-process → copy (3).", size=10, color=INK)
    save(fig, "10_three_roads.png")


# ----------------------------------------------------------------------------------------------
# 11. two torch datasets from one view
# ----------------------------------------------------------------------------------------------
def torch_two_ways():
    fig, ax = canvas(14.6, 6.6)
    cx = 7.3
    label(ax, cx, 6.3, "Road 2 has two shapes. The question is: what should one item be?", size=12, weight="bold")

    # view at top center
    box(ax, cx - 1.9, 4.75, 3.8, 1.05, FO, fc="#fff7f0", lw=1.6)
    label(ax, cx, 5.5, "view = dataset.take(10, seed=51)", size=9, family="DejaVu Sans Mono", color=INK)
    label(ax, cx, 5.08, "10 episode samples", size=8.8, color=MUTED)

    pw, ph, py = 6.8, 3.3, 1.05
    ys = py + 1.75

    def pipeline(x, color, stages):
        """stages: list of (w, fc, ec, line1, line2, c1, c2)"""
        xx = x + 0.3
        for i, (w, fc, ec, l1, l2, c1, c2) in enumerate(stages):
            box(ax, xx, ys - 0.3, w, 0.7, ec, fc=fc, lw=1.2 if fc != ec else 0)
            label(ax, xx + w / 2, ys + 0.2, l1, size=7.6, color=c1, weight="bold", family="DejaVu Sans Mono")
            label(ax, xx + w / 2, ys - 0.08, l2, size=7.0, family="DejaVu Sans Mono", color=c2)
            xx += w
            if i < len(stages) - 1:
                arrow(ax, xx + 0.05, ys + 0.05, xx + 0.4, ys + 0.05, color=color, lw=1.3, ms=10)
                xx += 0.45

    # left: to_torch
    lx = 0.3
    arrow(ax, cx - 1.9, 5.2, lx + pw / 2 + 1.2, py + ph + 0.05, color=BATCH, lw=1.8)
    box(ax, lx, py, pw, ph, BATCH, fc="white", lw=1.6)
    label(ax, lx + pw / 2, py + ph - 0.3, "6a   view.to_torch(EpisodeGetItem(A))", size=9.6, color=BATCH, weight="bold", family="DejaVu Sans Mono")
    label(ax, lx + pw / 2, py + ph - 0.65, "one item = one episode", size=9.5, color=INK, weight="bold")
    pipeline(lx, BATCH, [
        (1.95, "#fff7f0", FO, "required_keys", "media_reference, task", FO_DARK, INK),
        (2.05, "white", INK, "GetItem.__call__", "A[row0:row1] -> 25 steps", INK, INK),
        (1.75, BATCH, BATCH, "traj (25, 14)", "+ episode_index, task", "white", "white"),
    ])
    label(ax, lx + pw / 2, py + 0.95, "FiftyOneTorchDataset, len = 10", size=8.6, color=MUTED, family="DejaVu Sans Mono")
    label(ax, lx + pw / 2, py + 0.62, "DataLoader(batch_size=4, worker_init_fn=FiftyOneTorchDataset.worker_init)", size=7.4, color=MUTED, family="DejaVu Sans Mono")
    label(ax, lx + pw / 2, py + 0.28, "batch:  traj (4, 25, 14)         for episode-level models", size=8.6, color=INK, family="DejaVu Sans Mono")

    # right: LeRobotDataset(episodes=)
    rx = 14.6 - 0.3 - pw
    arrow(ax, cx + 1.9, 5.2, rx + pw / 2 - 1.2, py + ph + 0.05, color=ACTION, lw=1.8)
    box(ax, rx, py, pw, ph, ACTION, fc="white", lw=1.6)
    label(ax, rx + pw / 2, py + ph - 0.3, '6b   LeRobotDataset(root, episodes=view.values("episode_index"))', size=8.4, color=ACTION, weight="bold", family="DejaVu Sans Mono")
    label(ax, rx + pw / 2, py + ph - 0.65, "one item = one frame", size=9.5, color=INK, weight="bold")
    pipeline(rx, ACTION, [
        (2.2, "#fff7f0", FO, 'values("episode_index")', "[4, 8, 11, 18, 20, ...]", FO_DARK, INK),
        (1.8, "white", INK, "episodes=[...]", "+ delta_timestamps", INK, INK),
        (1.75, ACTION, ACTION, "Lesson 1 dict", "image, state, action", "white", "white"),
    ])
    label(ax, rx + pw / 2, py + 0.95, "LeRobotDataset, len = 5 000 frames", size=8.6, color=MUTED, family="DejaVu Sans Mono")
    label(ax, rx + pw / 2, py + 0.62, "DataLoader(batch_size=8)", size=7.4, color=MUTED, family="DejaVu Sans Mono")
    label(ax, rx + pw / 2, py + 0.28, "batch:  action (8, 10, 14) ...    for policies", size=8.6, color=INK, family="DejaVu Sans Mono")

    label(ax, cx, 0.5, "Both read the source folder FiftyOne pointed at. Nothing is exported or copied; FiftyOne's only job here was choosing the episodes.", size=9.5, color=MUTED)
    save(fig, "11_torch_two_ways.png")


# ==============================================================================================
# Lesson 3  (real data: A = actions, S = states, EP = episode index, all 25 000 rows)
# ==============================================================================================
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
L_GRIP, R_GRIP = 6, 13


def grasp_time(g, fps):
    mid = (g.max() + g.min()) / 2
    opened = np.where(g > mid)[0]
    if len(opened) == 0:
        return len(g) / fps
    closed = np.where(g[opened[0]:] < mid)[0]
    return (opened[0] + closed[0]) / fps if len(closed) else len(g) / fps


# ----------------------------------------------------------------------------------------------
# 12. from curves to six numbers on the sample
# ----------------------------------------------------------------------------------------------
def kinematics(A, S, EP, fps, ep=0):
    a, s = A[EP == ep], S[EP == ep]
    t = np.arange(len(a)) / fps
    step = np.linalg.norm(np.diff(a[:, ARM], axis=0), axis=1)
    tl, tr = grasp_time(a[:, L_GRIP], fps), grasp_time(a[:, R_GRIP], fps)
    feats = {
        "joint_travel": step.sum(), "peak_speed": step.max() * fps, "idle_frac": (step < 1e-3).mean(),
        "track_err": np.abs(a[:, ARM] - s[:, ARM]).mean(), "grasp_t_left": tl, "grasp_t_right": tr,
    }

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1.0], hspace=0.55, wspace=0.25, left=0.06, right=0.98, top=0.9, bottom=0.08)
    fig.suptitle(f"Episode {ep}: three views of the same 14 curves, reduced to six numbers on the FiftyOne sample", fontsize=12, color=INK, weight="bold", x=0.5, y=0.97)

    def style(ax, title):
        ax.set_title(title, fontsize=9.8, loc="left", color=INK)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(alpha=0.2)

    # (a) speed per step -> joint_travel, peak_speed, idle_frac
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(t[1:], step * fps, color=STATE, alpha=0.25)
    ax.plot(t[1:], step * fps, color=STATE, lw=1.2)
    k = step.argmax()
    ax.plot(t[1 + k], step[k] * fps, "o", color=PAD, ms=5)
    ax.annotate(f"peak_speed = {feats['peak_speed']:.2f}", (t[1 + k], step[k] * fps), xytext=(8, -2), textcoords="offset points", fontsize=8.5, color=PAD)
    ax.axhline(1e-3 * fps, color=MUTED, lw=0.8, ls="--")
    ax.text(t[-1], 1e-3 * fps, "  idle threshold", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    style(ax, f"joint speed per step, 12 arm joints    →  joint_travel = area = {feats['joint_travel']:.2f},   idle_frac = share below threshold = {feats['idle_frac']:.3f}")
    ax.set_ylabel("|Δaction| · fps", fontsize=8, color=MUTED)

    # (b) gripper curves -> grasp_t_left / right
    ax = fig.add_subplot(gs[1, 0])
    for g, tg, c, name in [(L_GRIP, tl, LEFT, "left"), (R_GRIP, tr, RIGHT, "right")]:
        ax.plot(t, a[:, g], color=c, lw=1.5, label=f"{name} gripper action")
        mid = (a[:, g].max() + a[:, g].min()) / 2
        ax.axhline(mid, color=c, lw=0.7, ls=":", alpha=0.7)
        ax.axvline(tg, color=c, lw=1.2, ls="--")
        ax.annotate(f"grasp_t_{name} = {tg:.2f} s", (tg, mid), xytext=(6, 14 if name == "left" else 22), textcoords="offset points", fontsize=8.5, color=c)
    style(ax, "gripper commands (high = open, low = closed)    →  first close after the first open")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (c) action vs state on one joint -> track_err
    ax = fig.add_subplot(gs[2, 0])
    j = 2
    ax.plot(t, a[:, j], color=ACTION, lw=1.4, label="action (commanded)")
    ax.plot(t, s[:, j], color=STATE, lw=1.4, label="state (measured)")
    ax.fill_between(t, a[:, j], s[:, j], color=PAD, alpha=0.3, label="|action − state|")
    style(ax, f"joint {j}: commanded vs measured    →  track_err = mean gap over 12 arm joints = {feats['track_err']:.4f}")
    ax.set_xlabel("seconds", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)

    # right: the sample card with the six fields
    ax = fig.add_subplot(gs[:, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.08, 0.06), 0.84, 0.88, boxstyle="round,pad=0.02,rounding_size=0.03", ec=FO, fc="#fff7f0", lw=1.6, transform=ax.transAxes))
    ax.text(0.5, 0.88, f"sample  (episode {ep})", ha="center", fontsize=10.5, color=FO_DARK, weight="bold", transform=ax.transAxes)
    y = 0.79
    for k_, v in [("episode_index", str(ep)), ("task", '"Insert the peg…"'), ("duration", "10.0")]:
        ax.text(0.14, y, k_, fontsize=8.5, family="DejaVu Sans Mono", color=MUTED, transform=ax.transAxes)
        ax.text(0.86, y, v, fontsize=8.5, family="DejaVu Sans Mono", color=MUTED, ha="right", transform=ax.transAxes)
        y -= 0.06
    ax.plot([0.14, 0.86], [y + 0.02, y + 0.02], color=LINE, lw=1, transform=ax.transAxes)
    ax.text(0.14, y - 0.035, "added by set_values", fontsize=8, color=FO_DARK, transform=ax.transAxes)
    y -= 0.1
    colors = {"joint_travel": STATE, "peak_speed": PAD, "idle_frac": STATE, "track_err": PAD, "grasp_t_left": LEFT, "grasp_t_right": RIGHT}
    for k_, v in feats.items():
        ax.text(0.14, y, k_, fontsize=9, family="DejaVu Sans Mono", color=colors[k_], weight="bold", transform=ax.transAxes)
        ax.text(0.86, y, f"{v:.3f}", fontsize=9, family="DejaVu Sans Mono", color=INK, ha="right", transform=ax.transAxes)
        y -= 0.068
    ax.text(0.5, 0.1, "one float per episode → sidebar sliders,\nsort_by, match, histograms", fontsize=8.3, color=MUTED, ha="center", transform=ax.transAxes)
    save(fig, "12_kinematics.png")


# ----------------------------------------------------------------------------------------------
# 13. temporal tags on the episode timeline
# ----------------------------------------------------------------------------------------------
def temporal_tags(A, EP, fps, ep=0):
    a = A[EP == ep]
    t = np.arange(len(a)) / fps
    tl, tr = grasp_time(a[:, L_GRIP], fps), grasp_time(a[:, R_GRIP], fps)
    NS = 1_000_000_000
    half = 0.25

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.08, left=0.06, right=0.98, top=0.86, bottom=0.12)
    fig.suptitle("A temporal tag is a [start, end) interval on one sample's timeline, in nanoseconds from the start of the episode",
                 fontsize=12, color=INK, weight="bold", y=0.97)

    # top: the App-style timeline with two tag bars
    ax0 = fig.add_subplot(gs[0]); ax0.set_xlim(0, 10); ax0.set_ylim(0, 1); ax0.axis("off")
    ax0.add_patch(Rectangle((0, 0.42), 10, 0.16, fc="#eceff1", ec="none"))
    for k in range(11):
        ax0.plot([k, k], [0.42, 0.58], color="white", lw=1.2)
        ax0.text(k, 0.3, f"{k} s", ha="center", fontsize=7.5, color=MUTED)
    for tg, c, name in [(tl, LEFT, "left grasp"), (tr, RIGHT, "right grasp")]:
        ax0.add_patch(Rectangle((tg - half, 0.42), 2 * half, 0.16, fc=c, ec="none", alpha=0.9))
        ax0.text(tg, 0.72, f'"{name}"', ha="center", fontsize=9, color=c, weight="bold")
        ax0.text(tg, 0.05, f"start = {int((tg - half) * NS):,} ns\nend   = {int((tg + half) * NS):,} ns", ha="center", fontsize=7.2, color=c, family="DejaVu Sans Mono")
    ax0.text(0, 0.88, "the App timeline for this episode", fontsize=9, color=MUTED)

    # bottom: the gripper curves the windows were derived from
    ax1 = fig.add_subplot(gs[1])
    for g, tg, c, name in [(L_GRIP, tl, LEFT, "left"), (R_GRIP, tr, RIGHT, "right")]:
        ax1.plot(t, a[:, g], color=c, lw=1.6, label=f"{name} gripper action")
        ax1.axvspan(tg - half, tg + half, color=c, alpha=0.18)
        ax1.axvline(tg, color=c, lw=1.0, ls="--")
    ax1.set_xlim(0, 10)
    ax1.set_xlabel("seconds", fontsize=9, color=MUTED)
    ax1.set_ylabel("gripper command", fontsize=9, color=MUTED)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["left"].set_color(LINE); ax1.spines["bottom"].set_color(LINE)
    ax1.tick_params(labelsize=8, colors=MUTED); ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax1.text(0.02, 0.06, 'dataset.temporal_tags.add(fota.TemporalTag(sample.id, start=…, end=…, tag="left grasp"))\n'
                         'dataset.match_temporal_tags(tags="right grasp", start=5 * NS)   →   episodes whose right grasp came after 5 s',
             transform=ax1.transAxes, fontsize=8.2, family="DejaVu Sans Mono", color=INK,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec=LINE))
    save(fig, "13_temporal_tags.png")


# ----------------------------------------------------------------------------------------------
# 14. motion embedding -> Brain
# ----------------------------------------------------------------------------------------------
def motion_embedding(A, EP, n_steps=25):
    eps = sorted(np.unique(EP))
    emb = np.stack([A[EP == e][np.linspace(0, (EP == e).sum() - 1, n_steps).astype(int)].ravel() for e in eps])
    # PCA by SVD, for the real map
    X = emb - emb.mean(axis=0)
    U, Sv, Vt = np.linalg.svd(X, full_matrices=False)
    pts = U[:, :2] * Sv[:2]
    dist = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    outliers = set(np.argsort(dist)[-5:])
    a0 = A[EP == 0]
    idx = np.linspace(0, len(a0) - 1, n_steps).astype(int)

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.35], height_ratios=[1, 1], wspace=0.35, hspace=0.45, left=0.05, right=0.98, top=0.86, bottom=0.08)
    fig.suptitle("The motion is the embedding: (500, 14) → 25 evenly spaced rows → one 350-dim vector per episode → Brain", fontsize=12, color=INK, weight="bold", y=0.96)

    def strip(ax):
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(labelsize=7.5, colors=MUTED)

    # (1) full matrix
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(a0, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    for i in idx:
        ax.axhline(i, color="black", lw=0.5, alpha=0.6)
    ax.set_title("episode 0 actions (500, 14)\nblack lines = the 25 rows kept", fontsize=9, loc="left", color=INK)
    ax.set_xticks([]); strip(ax); ax.set_ylabel("time", fontsize=8, color=MUTED)

    # (2) resampled
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(a0[idx], aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    ax.set_title("resampled (25, 14)", fontsize=9, loc="left", color=INK)
    ax.set_xticks(range(0, 14, 2)); ax.set_yticks([0, 12, 24]); strip(ax)

    # (3) flattened vector, all 50 stacked
    ax = fig.add_subplot(gs[:, 1])
    ax.imshow(emb, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2, interpolation="nearest")
    ax.set_title("ravel → embeddings (50, 350)\none row per episode", fontsize=9, loc="left", color=INK)
    ax.set_xlabel("350 = 25 steps × 14 joints", fontsize=8, color=MUTED); ax.set_ylabel("episode", fontsize=8, color=MUTED)
    ax.set_yticks([0, 10, 20, 30, 40, 49]); strip(ax)
    ax.axhline(0, color=INK, lw=1.5)

    # (4) similarity: neighbours of episode 0
    centred = emb - emb.mean(axis=0)                          # FiftyOne's sklearn backend: mean-centre, then cosine
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    d = 1 - unit @ unit[0]
    nn = np.argsort(d)[:6]
    ax = fig.add_subplot(gs[0, 2])
    ax.barh(range(6), d[nn][::-1], color=[BATCH] + [LINE] * 5 if False else [BATCH if e == 0 else MUTED for e in nn[::-1]])
    ax.set_yticks(range(6)); ax.set_yticklabels([f"episode {e}" for e in nn[::-1]], fontsize=8)
    ax.set_title("compute_similarity  →  sort_by_similarity(episode 0)\ndistance in embedding space", fontsize=9, loc="left", color=INK)
    ax.set_xlabel("cosine distance (mean-centred, as FiftyOne computes it)", fontsize=8, color=MUTED)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED)

    # (5) PCA map
    ax = fig.add_subplot(gs[1, 2])
    cols = [PAD if i in outliers else BATCH for i in range(len(eps))]
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=38, alpha=0.9)
    for i in outliers:
        ax.annotate(str(eps[i]), pts[i], xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=PAD)
    ax.scatter(*pts.mean(axis=0), marker="+", s=120, color=INK)
    ax.set_title('compute_visualization(method="pca")  →  2-D map\nred = 5 farthest from the centroid (+) → tag "traj_outlier"', fontsize=9, loc="left", color=INK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED); ax.grid(alpha=0.2)
    save(fig, "14_motion_embedding.png")


# ----------------------------------------------------------------------------------------------
# 15. the curation loop
# ----------------------------------------------------------------------------------------------
def curation_loop():
    fig, ax = canvas(14.6, 5.6)
    label(ax, 7.3, 5.3, "The loop this lesson runs four times: compute per episode \u2192 store on the sample \u2192 query \u2192 look \u2192 hand off", size=12, weight="bold")
    stages = [
        ("compute", "NumPy over A, S, EP:\nsix scalars,\ngrasp windows,\n350-dim motion vector", INK, "white"),
        ("store on samples", "set_values(field, vals)\ntemporal_tags.add(...)\ncompute_similarity(...)\ncompute_visualization(...)\ntag_samples(...)", FO, "#fff7f0"),
        ("query", "sort_by / match / limit\nmatch_temporal_tags(...)\nsort_by_similarity(...)\nsave_view(...)", FO, "#fff7f0"),
        ("look", "session.view = ...\nsidebar sliders\ntimeline markers\nEmbeddings panel", FO, "#fff7f0"),
        ("hand off", "view.export(\n  fo.types.LeRobotDataset)\nor LeRobotDataset(\n  episodes=view.values(...))", ACTION, "white"),
    ]
    w, h, gap, y = 2.55, 2.4, 0.3, 1.7
    x = 0.3
    for i, (title, body, color, fc) in enumerate(stages):
        box(ax, x, y, w, h, color, fc=fc, lw=1.6)
        label(ax, x + w / 2, y + h - 0.3, title, size=10.5, color=color, weight="bold")
        label(ax, x + w / 2, y + h / 2 - 0.2, body, size=7.7, family="DejaVu Sans Mono", color=INK)
        if i < len(stages) - 1:
            arrow(ax, x + w + 0.03, y + h / 2, x + w + gap - 0.03, y + h / 2, color=MUTED, lw=1.8, ms=14)
        x += w + gap
    arrow(ax, 0.3 + 3 * (w + gap) + w / 2, y - 0.05, 0.3 + w / 2, y - 0.05, color=MUTED, lw=1.2, ms=12, conn="arc3,rad=-0.25", ls=(0, (4, 3)))
    label(ax, 7.3, 0.42, "what you see in the App suggests the next thing to compute", size=8.8, color=MUTED)
    label(ax, 7.3, 0.05, "Everything in the orange boxes is stored on the dataset and survives the kernel: fields, tags, brain runs, saved views.", size=9.3, color=FO_DARK)
    save(fig, "15_curation_loop.png")


if __name__ == "__main__":
    dataset_table()
    one_frame()
    delta_timestamps()
    padding()
    batch()
    shape_flow()
    episode_sample()
    pointer_road()
    three_roads()
    torch_two_ways()
    curation_loop()

    # real data for the data-driven figures
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import logging
    logging.getLogger("lerobot").setLevel(logging.ERROR)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lerobot/aloha_sim_insertion_human")
    tbl = ds.hf_dataset.with_format("numpy")
    A, S, EP = (np.asarray(tbl[k]) for k in ["action", "observation.state", "episode_index"])
    episode_matrix(A[EP == 0], ds.fps)
    kinematics(A, S, EP, ds.fps)
    temporal_tags(A, EP, ds.fps)
    motion_embedding(A, EP)
