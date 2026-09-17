"""Render the concept diagrams used in lesson_01_native_lerobot_dataset.ipynb.

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


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6, style="-|>", ms=14, z=3, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z, linestyle=ls)
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


if __name__ == "__main__":
    dataset_table()
    one_frame()
    delta_timestamps()
    padding()
    batch()
    shape_flow()

    # real data for the episode matrix
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import logging
    logging.getLogger("lerobot").setLevel(logging.ERROR)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lerobot/aloha_sim_insertion_human", episodes=[0])
    tbl = ds.hf_dataset.with_format("numpy")
    episode_matrix(np.asarray(tbl["action"]), ds.fps)
