# pip install PyQt5 matplotlib numpy scipy
# Circuit_SmithChart.py — RF Circuit Design & Smith Chart Visualization Tool

import sys
import math
import uuid
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from scipy import linalg

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Arc, Circle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QProgressBar, QInputDialog,
    QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont


# =============================================================================
# Section 2: Data Models
# =============================================================================

GRID = 20
PIN_OFFSET = 30
PORT_X = 60

ERRORS = {
    'freq_empty':   "주파수 입력 오류\n시작/끝 주파수를 MHz 단위 숫자로 입력하세요.",
    'freq_order':   "주파수 범위 오류\n시작 주파수는 끝 주파수보다 작아야 합니다.",
    'no_component': "회로 없음\nPort에 연결된 소자가 없습니다.\n소자를 배치하고 Port와 연결하세요.",
    'no_value':     "값 미입력 소자가 있습니다.\n해당 소자(빨간 테두리)를 더블클릭하여 값을 입력하세요.",
    'isolated':     "고립된 소자가 있습니다.\nPort 신호선(+)에서 GND까지 경로에 포함되지 않는 소자(빨간 테두리)를 확인하세요.",
}


@dataclass
class Component:
    id: str
    type: str            # 'R', 'L', 'C'
    x: int
    y: int
    rotation: int = 0    # 0 or 90
    value: float = None
    selected: bool = False
    error_highlight: bool = False


@dataclass
class Wire:
    id: str
    start_comp_id: str   # component.id, 'PORT'
    start_pin: str       # 'left'/'right'/'top'/'bottom'/'plus'/'gnd'
    end_comp_id: str
    end_pin: str


# =============================================================================
# Section 3: Helper Functions
# =============================================================================

def snap_to_grid(val):
    return round(val / GRID) * GRID


def get_pin_positions(comp):
    """Return dict of pin_name -> (px, py) for a component."""
    cx, cy = comp.x, comp.y
    if comp.rotation == 0:
        return {
            'left': (cx - PIN_OFFSET, cy),
            'right': (cx + PIN_OFFSET, cy),
        }
    else:  # 90
        return {
            'top': (cx, cy - PIN_OFFSET),
            'bottom': (cx, cy + PIN_OFFSET),
        }


def get_port_pin_positions(canvas_height):
    mid_y = canvas_height // 2
    return {
        'plus': (PORT_X, mid_y - 40),
        'gnd': (PORT_X, mid_y + 40),
    }


def calc_wire_path(sx, sy, ex, ey):
    """L-shaped wire routing: horizontal first, then vertical."""
    if sy == ey:
        return [(sx, sy), (ex, ey)]
    return [(sx, sy), (ex, sy), (ex, ey)]


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def get_all_pin_ids(components, canvas_height):
    """Return list of all (comp_id, pin_name) tuples including Port."""
    pins = []
    pins.append(('PORT', 'plus'))
    pins.append(('PORT', 'gnd'))
    for comp in components:
        if comp.rotation == 0:
            pins.append((comp.id, 'left'))
            pins.append((comp.id, 'right'))
        else:
            pins.append((comp.id, 'top'))
            pins.append((comp.id, 'bottom'))
    return pins


def get_pin_pos(comp_id, pin_name, components, canvas_height):
    """Get pixel position of a specific pin."""
    if comp_id == 'PORT':
        port_pins = get_port_pin_positions(canvas_height)
        return port_pins[pin_name]
    for comp in components:
        if comp.id == comp_id:
            pins = get_pin_positions(comp)
            return pins[pin_name]
    return None


# =============================================================================
# Section 4: CircuitCanvas
# =============================================================================

class CircuitCanvas(QWidget):
    """Left panel: circuit editing area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 510)
        self.setFocusPolicy(Qt.StrongFocus)

        self.components = []
        self.wires = []
        self.placement_mode = None
        self.wiring_start = None  # (comp_id, pin_name)
        self.move_mode = False

        # Double-click disambiguation
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(150)
        self._click_timer.timeout.connect(self._handle_single_click)
        self._pending_click_event = None
        self._double_click_happened = False

    # ---- Paint ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor('#E8F5E9'))

        # Grid
        self._draw_grid(painter)

        # Port
        self._draw_port(painter)

        # Wires
        for wire in self.wires:
            self._draw_wire(painter, wire)

        # Components
        for comp in self.components:
            self._draw_component(painter, comp)

        # Wiring preview (cursor tracking not implemented — just highlight start pin)
        if self.wiring_start:
            cid, pname = self.wiring_start
            pos = get_pin_pos(cid, pname, self.components, self.height())
            if pos:
                painter.setPen(QPen(QColor('#FF9800'), 3))
                painter.setBrush(QColor('#FF9800'))
                painter.drawEllipse(int(pos[0]) - 5, int(pos[1]) - 5, 10, 10)

        painter.end()

    def _draw_grid(self, painter):
        pen = QPen(QColor('#BDBDBD'), 1, Qt.DotLine)
        painter.setPen(pen)
        for x in range(0, self.width(), GRID):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), GRID):
            painter.drawLine(0, y, self.width(), y)

    def _draw_port(self, painter):
        port_pins = get_port_pin_positions(self.height())
        px, py_plus = port_pins['plus']
        _, py_gnd = port_pins['gnd']

        # Vertical line
        painter.setPen(QPen(QColor('#212121'), 2))
        painter.drawLine(px, py_plus, px, py_gnd)

        # Plus pin (green circle)
        painter.setPen(QPen(QColor('#4CAF50'), 2))
        painter.setBrush(QColor('#4CAF50'))
        painter.drawEllipse(px - 5, py_plus - 5, 10, 10)

        # GND pin (green circle)
        painter.drawEllipse(px - 5, py_gnd - 5, 10, 10)

        # Labels
        painter.setPen(QPen(QColor('#212121'), 1))
        font = QFont('Arial', 9)
        painter.setFont(font)
        painter.drawText(px - 40, py_plus + 4, "Port")
        painter.drawText(px - 10, py_plus - 10, "+")
        painter.drawText(px - 16, py_gnd + 18, "GND")

    def _draw_wire(self, painter, wire):
        start_pos = get_pin_pos(wire.start_comp_id, wire.start_pin,
                                self.components, self.height())
        end_pos = get_pin_pos(wire.end_comp_id, wire.end_pin,
                              self.components, self.height())
        if not start_pos or not end_pos:
            return

        painter.setPen(QPen(QColor('#F44336'), 2))
        points = calc_wire_path(start_pos[0], start_pos[1],
                                end_pos[0], end_pos[1])
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]),
                             int(points[i + 1][0]), int(points[i + 1][1]))

        # Pin dots (red for connected)
        painter.setBrush(QColor('#F44336'))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(start_pos[0]) - 3, int(start_pos[1]) - 3, 6, 6)
        painter.drawEllipse(int(end_pos[0]) - 3, int(end_pos[1]) - 3, 6, 6)

    def _draw_component(self, painter, comp):
        cx, cy = comp.x, comp.y

        # Error highlight background
        if comp.error_highlight:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor('#FFEBEE'))
            if comp.rotation == 0:
                painter.drawRect(cx - 22, cy - 14, 44, 28)
            else:
                painter.drawRect(cx - 14, cy - 22, 28, 44)

        # Determine pen color
        if comp.error_highlight:
            pen = QPen(QColor('#D32F2F'), 3)
        elif comp.selected:
            pen = QPen(QColor('#1565C0'), 3)
        else:
            pen = QPen(QColor('#212121'), 2)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if comp.type == 'R':
            self._draw_resistor(painter, comp)
        elif comp.type == 'L':
            self._draw_inductor(painter, comp)
        elif comp.type == 'C':
            self._draw_capacitor(painter, comp)

        # Value label
        self._draw_value_label(painter, comp)

    def _draw_resistor(self, painter, comp):
        cx, cy = comp.x, comp.y
        if comp.rotation == 0:
            # Left lead
            painter.drawLine(cx - 30, cy, cx - 20, cy)
            # Zigzag
            path = QPainterPath()
            path.moveTo(cx - 20, cy)
            peaks = [
                (cx - 15, cy - 8), (cx - 10, cy + 8),
                (cx - 5, cy - 8), (cx, cy + 8),
                (cx + 5, cy - 8), (cx + 10, cy + 8),
                (cx + 15, cy - 8), (cx + 20, cy),
            ]
            for px, py in peaks:
                path.lineTo(px, py)
            painter.drawPath(path)
            # Right lead
            painter.drawLine(cx + 20, cy, cx + 30, cy)
        else:
            # Top lead
            painter.drawLine(cx, cy - 30, cx, cy - 20)
            # Zigzag (vertical)
            path = QPainterPath()
            path.moveTo(cx, cy - 20)
            peaks = [
                (cx - 8, cy - 15), (cx + 8, cy - 10),
                (cx - 8, cy - 5), (cx + 8, cy),
                (cx - 8, cy + 5), (cx + 8, cy + 10),
                (cx - 8, cy + 15), (cx, cy + 20),
            ]
            for px, py in peaks:
                path.lineTo(px, py)
            painter.drawPath(path)
            # Bottom lead
            painter.drawLine(cx, cy + 20, cx, cy + 30)

    def _draw_inductor(self, painter, comp):
        cx, cy = comp.x, comp.y
        if comp.rotation == 0:
            # Left lead
            painter.drawLine(cx - 30, cy, cx - 20, cy)
            # 4 semicircle arcs (upward bulge)
            arc_width = 10  # each arc spans 10px horizontally
            for i in range(4):
                ax = cx - 20 + i * arc_width
                rect_x = ax
                rect_y = cy - 10
                path = QPainterPath()
                path.moveTo(rect_x, cy)
                path.arcTo(rect_x, rect_y, arc_width, 10, 180, -180)
                painter.drawPath(path)
            # Right lead
            painter.drawLine(cx + 20, cy, cx + 30, cy)
        else:
            # Top lead
            painter.drawLine(cx, cy - 30, cx, cy - 20)
            # 4 semicircle arcs (left bulge)
            arc_height = 10
            for i in range(4):
                ay = cy - 20 + i * arc_height
                path = QPainterPath()
                path.moveTo(cx, ay)
                path.arcTo(cx - 10, ay, 10, arc_height, 90, 180)
                painter.drawPath(path)
            # Bottom lead
            painter.drawLine(cx, cy + 20, cx, cy + 30)

    def _draw_capacitor(self, painter, comp):
        cx, cy = comp.x, comp.y
        if comp.rotation == 0:
            # Left lead
            painter.drawLine(cx - 30, cy, cx - 4, cy)
            # Left plate
            pen_save = painter.pen()
            plate_pen = QPen(pen_save.color(), 3)
            painter.setPen(plate_pen)
            painter.drawLine(cx - 4, cy - 12, cx - 4, cy + 12)
            # Right plate
            painter.drawLine(cx + 4, cy - 12, cx + 4, cy + 12)
            painter.setPen(pen_save)
            # Right lead
            painter.drawLine(cx + 4, cy, cx + 30, cy)
        else:
            # Top lead
            painter.drawLine(cx, cy - 30, cx, cy - 4)
            # Top plate
            pen_save = painter.pen()
            plate_pen = QPen(pen_save.color(), 3)
            painter.setPen(plate_pen)
            painter.drawLine(cx - 12, cy - 4, cx + 12, cy - 4)
            # Bottom plate
            painter.drawLine(cx - 12, cy + 4, cx + 12, cy + 4)
            painter.setPen(pen_save)
            # Bottom lead
            painter.drawLine(cx, cy + 4, cx, cy + 30)

    def _draw_value_label(self, painter, comp):
        font = QFont('Arial', 9)
        painter.setFont(font)

        if comp.value is not None:
            unit = {'R': '\u03A9', 'L': 'nH', 'C': 'pF'}[comp.type]
            text = f"{comp.value:g}{unit}"
            painter.setPen(QPen(QColor('#212121'), 1))
        else:
            text = "?"
            painter.setPen(QPen(QColor('#9E9E9E'), 1))

        if comp.rotation == 0:
            painter.drawText(comp.x - 20, comp.y + 22, text)
        else:
            painter.drawText(comp.x + 14, comp.y + 4, text)

    # ---- Mouse Events ----

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self._double_click_happened = False
        self._pending_click_event = event
        self._click_timer.start()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self._double_click_happened = True
        self._click_timer.stop()

        mx, my = event.x(), event.y()
        for comp in self.components:
            dist = math.hypot(mx - comp.x, my - comp.y)
            if dist <= 30:
                self._show_value_dialog(comp)
                return

    def _handle_single_click(self):
        if self._double_click_happened:
            return

        event = self._pending_click_event
        if event is None:
            return

        mx, my = event.x(), event.y()

        # 1. Pin proximity check (8px) — highest priority
        pin_hit = self._find_pin_at(mx, my, 8)
        if pin_hit:
            comp_id, pin_name = pin_hit
            if self.wiring_start is None:
                # Start wiring
                self.wiring_start = (comp_id, pin_name)
                self.update()
                return
            else:
                # Complete wiring or cancel
                s_cid, s_pin = self.wiring_start
                if s_cid == comp_id and s_pin == pin_name:
                    # Same pin — cancel
                    self.wiring_start = None
                    self.update()
                    return
                # Create wire
                wire = Wire(
                    id=uuid.uuid4().hex[:8],
                    start_comp_id=s_cid,
                    start_pin=s_pin,
                    end_comp_id=comp_id,
                    end_pin=pin_name,
                )
                self.wires.append(wire)
                self.wiring_start = None
                self.update()
                return

        # Cancel wiring if clicking empty space
        if self.wiring_start:
            self.wiring_start = None
            self.update()
            return

        # 2. Placement mode
        if self.placement_mode:
            gx = snap_to_grid(mx)
            gy = snap_to_grid(my)
            # Block near Port
            if gx <= PORT_X + 40:
                return
            comp = Component(
                id=uuid.uuid4().hex[:8],
                type=self.placement_mode,
                x=gx,
                y=gy,
            )
            self.components.append(comp)
            self.placement_mode = None
            # Signal to parent to reset button styles
            parent = self.window()
            if hasattr(parent, '_reset_tool_buttons'):
                parent._reset_tool_buttons()
            self.update()
            return

        # 3. Move mode
        if self.move_mode:
            selected = [c for c in self.components if c.selected]
            if selected:
                comp = selected[0]
                gx = snap_to_grid(mx)
                gy = snap_to_grid(my)
                if gx > PORT_X + 40:
                    comp.x = gx
                    comp.y = gy
                    # Remove connected wires
                    self.wires = [w for w in self.wires
                                  if w.start_comp_id != comp.id
                                  and w.end_comp_id != comp.id]
            self.move_mode = False
            self.update()
            return

        # 4. Select component (30px radius)
        for comp in self.components:
            comp.selected = False

        for comp in self.components:
            dist = math.hypot(mx - comp.x, my - comp.y)
            if dist <= 30:
                comp.selected = True
                break

        self.update()

    def _find_pin_at(self, mx, my, radius):
        """Find a pin within radius pixels of (mx, my).
        Returns (comp_id, pin_name) or None."""
        # Check Port pins
        port_pins = get_port_pin_positions(self.height())
        for pname, (px, py) in port_pins.items():
            if math.hypot(mx - px, my - py) <= radius:
                return ('PORT', pname)

        # Check component pins
        for comp in self.components:
            pins = get_pin_positions(comp)
            for pname, (px, py) in pins.items():
                if math.hypot(mx - px, my - py) <= radius:
                    return (comp.id, pname)
        return None

    def _show_value_dialog(self, comp):
        label = {'R': '\u03A9', 'L': 'nH', 'C': 'pF'}[comp.type]
        hint = {'R': '예: 50', 'L': '예: 10', 'C': '예: 3.3'}[comp.type]
        val, ok = QInputDialog.getDouble(
            self, f'{comp.type} 값 입력',
            f'값을 입력하세요 (단위: {label})\n{hint}',
            value=comp.value or 0.0,
            min=1e-9, max=1e9, decimals=4
        )
        if ok:
            comp.value = val
            comp.error_highlight = False
            self.update()

    # ---- Keyboard Events ----

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Escape:
            self.wiring_start = None
            self.placement_mode = None
            parent = self.window()
            if hasattr(parent, '_reset_tool_buttons'):
                parent._reset_tool_buttons()
            self.update()
            return

        selected = [c for c in self.components if c.selected]
        if not selected:
            return
        comp = selected[0]

        if key == Qt.Key_R:
            comp.rotation = 90 if comp.rotation == 0 else 0
            # Remove connected wires (pin names change)
            self.wires = [w for w in self.wires
                          if w.start_comp_id != comp.id
                          and w.end_comp_id != comp.id]
            self.update()

        elif key == Qt.Key_Delete:
            self.wires = [w for w in self.wires
                          if w.start_comp_id != comp.id
                          and w.end_comp_id != comp.id]
            self.components.remove(comp)
            self.update()

        elif key == Qt.Key_M:
            self.move_mode = True


# =============================================================================
# Section 5: SmithChartCanvas
# =============================================================================

class SmithChartCanvas(FigureCanvasQTAgg):
    """Right panel: Smith Chart."""

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(5.5, 5.5))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setFixedSize(600, 510)

        self.Z0 = 50.0
        self._freqs = None
        self._Z_list = None
        self._gammas = None
        self._click_marker = None
        self._click_annotation = None

        self.draw_background()
        self.mpl_connect('button_press_event', self._on_click)

    def draw_background(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal')
        ax.axis('off')

        # Unit circle
        unit_circle = plt.Circle((0, 0), 1, fill=False,
                                 edgecolor='black', linewidth=1.5)
        ax.add_patch(unit_circle)

        # Real axis
        ax.plot([-1, 1], [0, 0], color='black', linewidth=1)

        # Constant resistance circles
        r_values = [0, 0.2, 0.5, 1, 2, 5]
        for r in r_values:
            center_x = r / (r + 1)
            radius = 1 / (r + 1)
            circle = plt.Circle((center_x, 0), radius, fill=False,
                                edgecolor='#9E9E9E', linewidth=0.5,
                                linestyle='--')
            circle.set_clip_path(plt.Circle((0, 0), 1,
                                            transform=ax.transData))
            ax.add_patch(circle)
            # Label
            label_x = (r - 1) / (r + 1) if r > 0 else -1
            if r == 0:
                ax.text(-1.0, 0.05, '0', fontsize=7, color='#757575')
            else:
                lx = r / (r + 1) - 1 / (r + 1)
                ax.text(lx, 0.05, str(r), fontsize=7, color='#757575')

        # Constant reactance arcs
        x_values = [0.2, 0.5, 1, 2, 5]
        for x in x_values:
            for sign in [1, -1]:
                xv = x * sign
                center_y = 1.0 / xv
                radius = abs(1.0 / xv)

                # Draw arc clipped to unit circle
                theta_points = np.linspace(0, 2 * np.pi, 500)
                arc_x = 1.0 + radius * np.cos(theta_points)
                arc_y = center_y + radius * np.sin(theta_points)

                # Clip to unit circle
                mask = arc_x ** 2 + arc_y ** 2 <= 1.01
                arc_x = np.where(mask, arc_x, np.nan)
                arc_y = np.where(mask, arc_y, np.nan)

                ax.plot(arc_x, arc_y, color='#9E9E9E', linewidth=0.5,
                        linestyle=':')

        self.draw()

    def plot_results(self, freqs_MHz, Z_list):
        self._freqs = freqs_MHz
        self._Z_list = Z_list

        self.draw_background()
        ax = self.ax

        Z0 = self.Z0
        gammas = []
        for Z in Z_list:
            if Z == complex('inf') or abs(Z + Z0) < 1e-30:
                gammas.append(complex(1, 0))
            else:
                g = (Z - Z0) / (Z + Z0)
                # Clip to unit circle
                if abs(g) > 1:
                    g = g / abs(g)
                gammas.append(g)

        self._gammas = gammas
        xs = [g.real for g in gammas]
        ys = [g.imag for g in gammas]

        # Main line
        ax.plot(xs, ys, color='#1565C0', linewidth=2, zorder=5)

        # Start marker
        ax.plot(xs[0], ys[0], marker='>', color='#1565C0',
                markersize=8, zorder=6)
        ax.annotate(f"{freqs_MHz[0]:.0f}MHz", (xs[0], ys[0]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color='#1565C0')

        # End marker
        ax.plot(xs[-1], ys[-1], marker='s', color='#1565C0',
                markersize=8, zorder=6)
        ax.annotate(f"{freqs_MHz[-1]:.0f}MHz", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color='#1565C0')

        # 13.56 MHz special marker
        if freqs_MHz[0] <= 13.56 <= freqs_MHz[-1]:
            idx = int(np.argmin(np.abs(np.array(freqs_MHz) - 13.56)))
            ax.plot(xs[idx], ys[idx], marker='s', color='#E53935',
                    markersize=8, zorder=6)
            ax.annotate("13.56MHz", (xs[idx], ys[idx]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color='#E53935')

        self._click_marker = None
        self._click_annotation = None
        self.draw()

    def _on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self._gammas is None:
            return

        # Find nearest point in display coordinates
        click_disp = self.ax.transData.transform((event.xdata, event.ydata))
        min_dist = float('inf')
        min_idx = -1

        for i, g in enumerate(self._gammas):
            pt_disp = self.ax.transData.transform((g.real, g.imag))
            dist = math.hypot(pt_disp[0] - click_disp[0],
                              pt_disp[1] - click_disp[1])
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_dist > 20 or min_idx < 0:
            return

        # Remove previous click marker
        if self._click_marker:
            try:
                self._click_marker.remove()
            except ValueError:
                pass
        if self._click_annotation:
            try:
                self._click_annotation.remove()
            except ValueError:
                pass

        g = self._gammas[min_idx]
        Z = self._Z_list[min_idx]
        freq = self._freqs[min_idx]

        self._click_marker = self.ax.plot(
            g.real, g.imag, 'o', color='#FF6F00', markersize=10, zorder=7
        )[0]

        label = f"f = {freq:.2f}MHz\nZ = {Z.real:.2f} + j{Z.imag:.2f} \u03A9"
        self._click_annotation = self.ax.annotate(
            label, (g.real, g.imag),
            textcoords="offset points", xytext=(10, 10),
            fontsize=8, color='#FF6F00',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#FF6F00', alpha=0.9),
            zorder=8
        )

        self.draw()


# =============================================================================
# Section 6: CalcWorker (QThread) with MNA Engine
# =============================================================================

class CalcWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(self, f_start_mhz, f_end_mhz, components, wires,
                 canvas_height, parent=None):
        super().__init__(parent)
        self.f_start = f_start_mhz
        self.f_end = f_end_mhz
        self.components = components
        self.wires = wires
        self.canvas_height = canvas_height

    def run(self):
        try:
            component_nodes, num_nodes = self._build_circuit_graph()

            step = 0.01  # MHz
            freqs = np.arange(self.f_start, self.f_end + step / 2, step)
            results = []

            for i, f_mhz in enumerate(freqs):
                f_hz = f_mhz * 1e6
                Z = self._solve_mna(f_hz, component_nodes, num_nodes)
                results.append(Z)
                if i % 100 == 0:
                    pct = int(i / len(freqs) * 100)
                    self.progress.emit(pct)

            self.progress.emit(100)
            self.result.emit(freqs.tolist(), results)

        except Exception as e:
            self.error.emit(str(e))

    def _build_circuit_graph(self):
        """Build node mapping from wires using Union-Find."""
        uf = UnionFind()

        # Create pin identifiers for all pins
        # Port pins
        uf.make_set(('PORT', 'plus'))
        uf.make_set(('PORT', 'gnd'))

        for comp in self.components:
            if comp.rotation == 0:
                uf.make_set((comp.id, 'left'))
                uf.make_set((comp.id, 'right'))
            else:
                uf.make_set((comp.id, 'top'))
                uf.make_set((comp.id, 'bottom'))

        # Union connected pins via wires
        for wire in self.wires:
            pin_a = (wire.start_comp_id, wire.start_pin)
            pin_b = (wire.end_comp_id, wire.end_pin)
            uf.make_set(pin_a)
            uf.make_set(pin_b)
            uf.union(pin_a, pin_b)

        # Assign node IDs
        # GND node = 0, Port plus = 1
        gnd_root = uf.find(('PORT', 'gnd'))
        plus_root = uf.find(('PORT', 'plus'))

        node_map = {}
        node_map[gnd_root] = 0
        next_node = 1
        if plus_root != gnd_root:
            node_map[plus_root] = next_node
            next_node += 1

        # Assign nodes to all pins
        for key in uf.parent:
            root = uf.find(key)
            if root not in node_map:
                node_map[root] = next_node
                next_node += 1

        # Build component_nodes: {comp_id: (node_a, node_b)}
        component_nodes = {}
        for comp in self.components:
            if comp.rotation == 0:
                pin_a = (comp.id, 'left')
                pin_b = (comp.id, 'right')
            else:
                pin_a = (comp.id, 'top')
                pin_b = (comp.id, 'bottom')
            node_a = node_map[uf.find(pin_a)]
            node_b = node_map[uf.find(pin_b)]
            component_nodes[comp.id] = (node_a, node_b)

        return component_nodes, next_node

    def _solve_mna(self, freq_hz, component_nodes, num_nodes):
        """Solve MNA for a single frequency. Returns port impedance Z."""
        N = num_nodes - 1  # Exclude GND (node 0)
        if N <= 0:
            return complex('inf')

        G = np.zeros((N, N), dtype=complex)
        I = np.zeros(N, dtype=complex)

        omega = 2 * math.pi * freq_hz

        for comp in self.components:
            na, nb = component_nodes[comp.id]
            val = comp.value

            if comp.type == 'R':
                Y = 1.0 / val
            elif comp.type == 'L':
                L_henry = val * 1e-9  # nH to H
                if abs(omega * L_henry) < 1e-30:
                    Y = complex('inf')
                else:
                    Y = 1.0 / (1j * omega * L_henry)
            elif comp.type == 'C':
                C_farad = val * 1e-12  # pF to F
                Y = 1j * omega * C_farad
            else:
                continue

            # Admittance stamping (node 0 = GND excluded from matrix)
            a = na - 1  # matrix index (0-based, GND skipped)
            b = nb - 1
            if a >= 0:
                G[a, a] += Y
            if b >= 0:
                G[b, b] += Y
            if a >= 0 and b >= 0:
                G[a, b] -= Y
                G[b, a] -= Y

        # Current injection: 1A at Port PLUS (node 1 -> matrix index 0)
        I[0] = 1.0

        try:
            V = linalg.solve(G, I)
            Z_port = V[0] / 1.0  # V at port plus / 1A
            return Z_port
        except (linalg.LinAlgError, np.linalg.LinAlgError):
            return complex('inf')


# =============================================================================
# Section 7: MainWindow
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Circuit Design & Smith Chart")
        self.setFixedSize(1200, 600)

        self._calc_worker = None

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- Top toolbar ----
        toolbar = QWidget()
        toolbar.setFixedHeight(50)
        toolbar.setStyleSheet("background-color: #2196F3;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)

        self._btn_r = QPushButton("R")
        self._btn_l = QPushButton("L")
        self._btn_c = QPushButton("C")

        for btn in [self._btn_r, self._btn_l, self._btn_c]:
            btn.setFixedSize(50, 36)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #1565C0;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #E3F2FD;
                }
            """)

        self._btn_r.clicked.connect(lambda: self._set_placement('R'))
        self._btn_l.clicked.connect(lambda: self._set_placement('L'))
        self._btn_c.clicked.connect(lambda: self._set_placement('C'))

        toolbar_layout.addWidget(self._btn_r)
        toolbar_layout.addWidget(self._btn_l)
        toolbar_layout.addWidget(self._btn_c)
        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar)

        # ---- Content area ----
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Left panel (circuit + controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.circuit_canvas = CircuitCanvas()
        left_layout.addWidget(self.circuit_canvas)

        # Bottom controls
        controls = QWidget()
        controls.setFixedHeight(40)
        controls.setStyleSheet("background-color: #F5F5F5;")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(8, 4, 8, 4)

        controls_layout.addWidget(QLabel("Freq"))
        self._freq_start = QLineEdit()
        self._freq_start.setFixedWidth(70)
        self._freq_start.setPlaceholderText("시작")
        controls_layout.addWidget(self._freq_start)

        controls_layout.addWidget(QLabel("to"))
        self._freq_end = QLineEdit()
        self._freq_end.setFixedWidth(70)
        self._freq_end.setPlaceholderText("끝")
        controls_layout.addWidget(self._freq_end)

        controls_layout.addWidget(QLabel("MHz"))

        self._btn_cal = QPushButton("Cal")
        self._btn_cal.setFixedWidth(50)
        self._btn_cal.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self._btn_cal.clicked.connect(self._on_cal_clicked)
        controls_layout.addWidget(self._btn_cal)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(120)
        self._progress.setVisible(False)
        controls_layout.addWidget(self._progress)

        controls_layout.addStretch()

        left_layout.addWidget(controls)
        left_panel.setFixedWidth(600)

        content_layout.addWidget(left_panel)

        # Right panel (Smith Chart)
        self.smith_chart = SmithChartCanvas()
        content_layout.addWidget(self.smith_chart)

        main_layout.addWidget(content)

    def _set_placement(self, comp_type):
        self._reset_tool_buttons()
        self.circuit_canvas.placement_mode = comp_type
        btn = {'R': self._btn_r, 'L': self._btn_l, 'C': self._btn_c}[comp_type]
        btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                border: 1px solid #E65100;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                color: white;
            }
        """)

    def _reset_tool_buttons(self):
        self.circuit_canvas.placement_mode = None
        for btn in [self._btn_r, self._btn_l, self._btn_c]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #1565C0;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #E3F2FD;
                }
            """)

    def _on_cal_clicked(self):
        canvas = self.circuit_canvas
        # Clear previous error highlights
        for comp in canvas.components:
            comp.error_highlight = False

        err = self._validate_circuit()
        if err:
            QMessageBox.warning(self, "오류", ERRORS[err])
            canvas.update()
            return

        f_start = float(self._freq_start.text())
        f_end = float(self._freq_end.text())

        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._btn_cal.setEnabled(False)

        self._calc_worker = CalcWorker(
            f_start, f_end,
            canvas.components, canvas.wires,
            canvas.height()
        )
        self._calc_worker.progress.connect(self._progress.setValue)
        self._calc_worker.result.connect(self._on_calc_result)
        self._calc_worker.error.connect(self._on_calc_error)
        self._calc_worker.start()

    def _on_calc_result(self, freqs, Z_list):
        self._progress.setVisible(False)
        self._btn_cal.setEnabled(True)
        self.smith_chart.plot_results(freqs, Z_list)

    def _on_calc_error(self, msg):
        self._progress.setVisible(False)
        self._btn_cal.setEnabled(True)
        QMessageBox.warning(self, "계산 오류", msg)

    def _validate_circuit(self):
        canvas = self.circuit_canvas

        # 1. Frequency input validation
        try:
            f_start = float(self._freq_start.text())
            f_end = float(self._freq_end.text())
        except (ValueError, TypeError):
            return 'freq_empty'

        if f_start <= 0 or f_end <= 0:
            return 'freq_empty'

        if f_start >= f_end:
            return 'freq_order'

        # 2. No components
        if not canvas.components:
            return 'no_component'

        # Check if any component is connected to Port
        port_connected = False
        for wire in canvas.wires:
            if wire.start_comp_id == 'PORT' or wire.end_comp_id == 'PORT':
                port_connected = True
                break
        if not port_connected:
            return 'no_component'

        # 3. Missing values
        missing = [c for c in canvas.components if c.value is None]
        if missing:
            for c in missing:
                c.error_highlight = True
            return 'no_value'

        # 4. Isolated component check (BFS from Port plus to Port gnd)
        # Build adjacency: pin -> set of connected pins (via wires)
        adj = {}
        for wire in canvas.wires:
            a = (wire.start_comp_id, wire.start_pin)
            b = (wire.end_comp_id, wire.end_pin)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        # Also connect the two pins of each component internally
        for comp in canvas.components:
            if comp.rotation == 0:
                pa = (comp.id, 'left')
                pb = (comp.id, 'right')
            else:
                pa = (comp.id, 'top')
                pb = (comp.id, 'bottom')
            adj.setdefault(pa, set()).add(pb)
            adj.setdefault(pb, set()).add(pa)

        # BFS from Port plus
        visited = set()
        queue = deque([('PORT', 'plus')])
        visited.add(('PORT', 'plus'))

        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check GND reachable
        if ('PORT', 'gnd') not in visited:
            # Mark all components as isolated
            for comp in canvas.components:
                comp.error_highlight = True
            return 'isolated'

        # Check each component is visited
        visited_comp_ids = {pin[0] for pin in visited if pin[0] != 'PORT'}
        isolated = [c for c in canvas.components if c.id not in visited_comp_ids]
        if isolated:
            for c in isolated:
                c.error_highlight = True
            return 'isolated'

        return None


# =============================================================================
# Section 8: Entry Point
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
