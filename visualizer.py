"""
TSP Visualization Module
Modern, high-contrast animations for TSP search processes.

Key features requested:
- No city index labels during search (optional switch).
- Always show the shared initial route as a dashed baseline.
- Stop at the end (no looping / no restart).
- Single animation that can show all three algorithms at once.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TSPVisualizer:
    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must be an (n, 2) array")
        self.n_cities = self.coords.shape[0]

    def _tour_xy(self, tour: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return x,y arrays that close the tour loop."""
        idx = np.asarray(tour + [tour[0]], dtype=int)
        xy = self.coords[idx]
        return xy[:, 0], xy[:, 1]

    def _try_fullscreen(self, fig, fullscreen: bool) -> None:
        if not fullscreen:
            return
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, "full_screen_toggle"):
                manager.full_screen_toggle()
        except Exception:
            pass

    def animate_single_algorithm(
        self,
        history: List[Tuple[List[int], float]],
        title: str,
        color: str = "#4A90E2",
        interval: int = 50,
        save_path: Optional[str] = None,
        initial_tour: Optional[List[int]] = None,
        show_city_labels: bool = False,
        fullscreen: bool = True,
    ):
        """
        Animate one algorithm.

        Parameters
        ----------
        history:
            List of (tour, length) entries (best-so-far each step).
        initial_tour:
            The shared random starting tour. If None, uses history[0][0].
        """

        if not history:
            raise ValueError("history is empty")

        if initial_tour is None:
            initial_tour = list(history[0][0])

        distances = np.array([h[1] for h in history], dtype=float)

        # Figure / layout
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        self._try_fullscreen(fig, fullscreen)
        fig.patch.set_facecolor("#0b1020")

        gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1.0], height_ratios=[3.0, 1.0])

        ax_main = fig.add_subplot(gs[:, 0])
        ax_prog = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[1, 1])

        # ---- Main axis styling
        ax_main.set_facecolor("#0b1020")
        ax_main.set_title(title, fontsize=16, fontweight="bold", color=color, pad=14)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        for sp in ax_main.spines.values():
            sp.set_visible(False)

        # City glow + core
        ax_main.scatter(self.coords[:, 0], self.coords[:, 1], s=260, alpha=0.08, edgecolors="none")
        ax_main.scatter(self.coords[:, 0], self.coords[:, 1], s=36, alpha=0.95, edgecolors="white", linewidths=0.8)

        if show_city_labels:
            for i, (x, y) in enumerate(self.coords):
                ax_main.text(x, y, str(i), fontsize=7, ha="center", va="center", color="white", alpha=0.9)

        # Baseline: initial tour (dashed)
        x0, y0 = self._tour_xy(initial_tour)
        base_line, = ax_main.plot(
            x0, y0,
            linestyle="--",
            linewidth=1.4,
            color="#9aa4b2",
            alpha=0.45,
            zorder=1,
        )

        # Current tour (glow + core)
        glow_line, = ax_main.plot([], [], linewidth=7.0, alpha=0.10, color=color, zorder=2)
        tour_line, = ax_main.plot([], [], linewidth=2.6, alpha=0.95, color=color, zorder=3)

        # ---- Progress axis
        ax_prog.set_facecolor("#0b1020")
        ax_prog.set_title("Distance over time", fontsize=12, fontweight="bold", color="#cbd5e1", pad=10)
        ax_prog.set_xlabel("Step", fontsize=9, color="#cbd5e1")
        ax_prog.set_ylabel("Distance", fontsize=9, color="#cbd5e1")
        ax_prog.tick_params(colors="#cbd5e1", labelsize=8)
        for sp in ax_prog.spines.values():
            sp.set_color("#334155")
        ax_prog.grid(True, alpha=0.15, linestyle="--")

        prog_line, = ax_prog.plot([], [], linewidth=2.0, alpha=0.9, color=color)
        prog_dot, = ax_prog.plot([], [], marker="o", markersize=6, alpha=0.9, color=color)

        ax_prog.set_xlim(0, max(1, len(history) - 1))
        ymin = float(np.nanmin(distances)) * 0.985
        ymax = float(np.nanmax(distances)) * 1.015
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 1.0
        ax_prog.set_ylim(ymin, ymax)

        # ---- Info axis
        ax_info.set_facecolor("#0b1020")
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for sp in ax_info.spines.values():
            sp.set_visible(False)

        info_title = ax_info.text(0.02, 0.82, "Status", fontsize=12, fontweight="bold", color="#cbd5e1")
        info_step = ax_info.text(0.02, 0.58, "", fontsize=11, color="#cbd5e1")
        info_dist = ax_info.text(0.02, 0.35, "", fontsize=11, color=color, fontweight="bold")
        info_note = ax_info.text(0.02, 0.12, "Dashed line = shared initial tour", fontsize=9, color="#94a3b8")

        # ---- Animation funcs
        def init():
            glow_line.set_data([], [])
            tour_line.set_data([], [])
            prog_line.set_data([], [])
            prog_dot.set_data([], [])
            info_step.set_text("")
            info_dist.set_text("")
            return glow_line, tour_line, prog_line, prog_dot, info_step, info_dist, info_title, info_note, base_line

        best_so_far = distances[0]

        def animate(frame: int):
            nonlocal best_so_far

            tour, dist = history[frame]
            x, y = self._tour_xy(tour)
            glow_line.set_data(x, y)
            tour_line.set_data(x, y)

            xs = np.arange(frame + 1)
            ys = distances[: frame + 1]
            prog_line.set_data(xs, ys)
            prog_dot.set_data([frame], [dist])

            # Simple improvement indicator (ASCII only)
            if frame == 0:
                status = "Starting from shared random tour"
            else:
                if dist < best_so_far:
                    status = "Improved"
                    best_so_far = dist
                else:
                    status = "Searching / no improvement"

            info_step.set_text(f"Step: {frame} / {len(history) - 1}  |  {status}")
            info_dist.set_text(f"Best distance: {dist:.2f}")

            if frame == len(history) - 1:
                info_step.set_text(f"Finished (stagnation stop). Steps: {len(history) - 1}")
                info_dist.set_text(f"Final best distance: {distances[-1]:.2f}")

            return glow_line, tour_line, prog_line, prog_dot, info_step, info_dist, info_title, info_note, base_line

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(history),
            interval=interval,
            blit=True,
            repeat=False,
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=max(1, int(1000 / max(1, interval))))

        plt.show()
        return anim

    def animate_comparison(
        self,
        histories: Dict[str, List[Tuple[List[int], float]]],
        colors: Optional[Dict[str, str]] = None,
        interval: int = 50,
        save_path: Optional[str] = None,
        initial_tour: Optional[List[int]] = None,
        show_city_labels: bool = False,
        fullscreen: bool = True,
    ):
        """
        Animate multiple algorithms in ONE figure.

        The main panel overlays each algorithm's best-so-far tour with a different color.
        The right-top panel shows distance trajectories for all algorithms.
        """

        if not histories:
            raise ValueError("histories is empty")

        names = list(histories.keys())
        if colors is None:
            # Pleasant default palette
            palette = ["#FF6B6B", "#4ECDC4", "#7C5CFC", "#F7B801"]
            colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}

        # Determine shared initial tour
        if initial_tour is None:
            # Use the first history's first tour
            initial_tour = list(histories[names[0]][0][0])

        # Determine max length (frames)
        max_len = max(len(h) for h in histories.values())
        if max_len == 0:
            raise ValueError("At least one history is empty")

        def get_state(name: str, frame: int) -> Tuple[List[int], float]:
            h = histories[name]
            if not h:
                raise ValueError(f"Empty history for {name}")
            if frame < len(h):
                return h[frame][0], float(h[frame][1])
            return h[-1][0], float(h[-1][1])

        # Precompute distance arrays (padded with last value)
        dist_series: Dict[str, np.ndarray] = {}
        for name in names:
            d = np.array([x[1] for x in histories[name]], dtype=float)
            if len(d) < max_len:
                d = np.pad(d, (0, max_len - len(d)), mode="edge")
            dist_series[name] = d

        # y-limits from all series
        all_d = np.concatenate([dist_series[n] for n in names])
        ymin = float(np.nanmin(all_d)) * 0.985
        ymax = float(np.nanmax(all_d)) * 1.015
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 1.0

        # Figure / layout
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        self._try_fullscreen(fig, fullscreen)
        fig.patch.set_facecolor("#0b1020")

        gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.0], height_ratios=[3.0, 1.0])
        ax_main = fig.add_subplot(gs[:, 0])
        ax_prog = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[1, 1])

        # ---- Main axis
        ax_main.set_facecolor("#0b1020")
        ax_main.set_title("Algorithm comparison (same initial tour)", fontsize=16, fontweight="bold", color="#e2e8f0", pad=14)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        for sp in ax_main.spines.values():
            sp.set_visible(False)

        ax_main.scatter(self.coords[:, 0], self.coords[:, 1], s=260, alpha=0.08, edgecolors="none")
        ax_main.scatter(self.coords[:, 0], self.coords[:, 1], s=36, alpha=0.95, edgecolors="white", linewidths=0.8)

        if show_city_labels:
            for i, (x, y) in enumerate(self.coords):
                ax_main.text(x, y, str(i), fontsize=7, ha="center", va="center", color="white", alpha=0.9)

        x0, y0 = self._tour_xy(initial_tour)
        base_line, = ax_main.plot(x0, y0, linestyle="--", linewidth=1.4, color="#9aa4b2", alpha=0.45, zorder=1)

        # One (glow+core) line per algorithm
        glow_lines = {}
        core_lines = {}
        for name in names:
            c = colors[name]
            glow, = ax_main.plot([], [], linewidth=7.0, alpha=0.10, color=c, zorder=2)
            core, = ax_main.plot([], [], linewidth=2.4, alpha=0.92, color=c, zorder=3, label=name)
            glow_lines[name] = glow
            core_lines[name] = core

        leg = ax_main.legend(loc="upper right", frameon=False, fontsize=10)
        for t in leg.get_texts():
            t.set_color("#e2e8f0")

        # ---- Progress axis
        ax_prog.set_facecolor("#0b1020")
        ax_prog.set_title("Distance over time", fontsize=12, fontweight="bold", color="#cbd5e1", pad=10)
        ax_prog.set_xlabel("Step", fontsize=9, color="#cbd5e1")
        ax_prog.set_ylabel("Distance", fontsize=9, color="#cbd5e1")
        ax_prog.tick_params(colors="#cbd5e1", labelsize=8)
        for sp in ax_prog.spines.values():
            sp.set_color("#334155")
        ax_prog.grid(True, alpha=0.15, linestyle="--")
        ax_prog.set_xlim(0, max_len - 1)
        ax_prog.set_ylim(ymin, ymax)

        prog_lines = {}
        prog_dots = {}
        for name in names:
            c = colors[name]
            line, = ax_prog.plot([], [], linewidth=2.0, alpha=0.9, color=c)
            dot, = ax_prog.plot([], [], marker="o", markersize=5, alpha=0.9, color=c)
            prog_lines[name] = line
            prog_dots[name] = dot

        # ---- Info axis
        ax_info.set_facecolor("#0b1020")
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for sp in ax_info.spines.values():
            sp.set_visible(False)

        info_title = ax_info.text(0.02, 0.82, "Live summary", fontsize=12, fontweight="bold", color="#cbd5e1")
        info_step = ax_info.text(0.02, 0.60, "", fontsize=11, color="#cbd5e1")
        info_lines = {}
        y = 0.40
        for name in names:
            info_lines[name] = ax_info.text(0.02, y, "", fontsize=10.5, color=colors[name], fontweight="bold")
            y -= 0.16

        def init():
            for name in names:
                glow_lines[name].set_data([], [])
                core_lines[name].set_data([], [])
                prog_lines[name].set_data([], [])
                prog_dots[name].set_data([], [])
                info_lines[name].set_text("")
            info_step.set_text("")
            return (
                [base_line, info_title, info_step]
                + list(glow_lines.values())
                + list(core_lines.values())
                + list(prog_lines.values())
                + list(prog_dots.values())
                + list(info_lines.values())
            )

        def animate(frame: int):
            # Update each algorithm overlay
            for name in names:
                tour, dist = get_state(name, frame)
                x, y = self._tour_xy(tour)
                glow_lines[name].set_data(x, y)
                core_lines[name].set_data(x, y)

                xs = np.arange(frame + 1)
                ys = dist_series[name][: frame + 1]
                prog_lines[name].set_data(xs, ys)
                prog_dots[name].set_data([frame], [ys[-1]])
                info_lines[name].set_text(f"{name}: {ys[-1]:.2f}")

            info_step.set_text(f"Step: {frame} / {max_len - 1}")

            if frame == max_len - 1:
                info_step.set_text("Finished (all animations ended, no looping).")

            artists = (
                [base_line, info_title, info_step]
                + list(glow_lines.values())
                + list(core_lines.values())
                + list(prog_lines.values())
                + list(prog_dots.values())
                + list(info_lines.values())
            )
            return artists

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=max_len,
            interval=interval,
            blit=True,
            repeat=False,
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=max(1, int(1000 / max(1, interval))))

        plt.show()
        return anim
