# -*- coding: utf-8 -*-
"""
v70 PF ビューア — 複数PF(9本)の軌跡 × 真値 × oracle(各行どのPFが最良か)を可視化する GUIアプリ。
=====================================================================================
- 左: 坑井リスト(硬い順=median RMSE 降順。検索可)。右: 軌跡プロット + oracleストリップ + 統計。
- 上プロット: 9本のPF(pf_ancc_1..9)を薄色, 真値TVTを太線, **真値上に「その行のoracleバンク色」の点**。
- 下ストリップ: 各行の oracle バンク番号(どのPFが勝ってるか)を色帯で表示。
- 統計: 各PFの overall RMSE / median / oracle。
データ: v70/artifacts_v70/multipf_s{stride}.pkl(=build_multipf_v70) + v68 cache(真値)。
実行: EVAL_STRIDE=4 .venv/Scripts/python.exe v70/pf_viewer_v70.py
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
STR = int(os.environ.get("EVAL_STRIDE", "4"))
CACHE = PROJ / "v68" / "artifacts_v68" / f"cache_v68_s{STR}.pkl"
PFP = PROJ / "v70" / "artifacts_v70" / f"multipf_s{STR}.pkl"

pg.setConfigOptions(antialias=True, background="k", foreground="w")

# 9バンク用の見分けやすい色
PALETTE = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
           "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff"]


def load():
    cache = {x["wid"]: np.asarray(x["target"], float) for x in pickle.load(open(CACHE, "rb"))}
    pf = pickle.loads(PFP.read_bytes())
    names = pf[next(iter(pf))]["names"]; K = len(names)
    stats = {}
    for wid, d in pf.items():
        if wid not in cache: continue
        A = np.asarray(d["ancc"], float); t = cache[wid]; n = min(A.shape[1], len(t)); A = A[:, :n]; t = t[:n]
        err = A - t[None, :]
        with np.errstate(invalid="ignore"):
            per = np.sqrt(np.nanmean(err ** 2, 1))
            med = float(np.sqrt(np.nanmean((np.nanmedian(A, 0) - t) ** 2)))
            oidx = np.nanargmin(np.abs(err), 0).astype(int)
            orc = float(np.sqrt(np.nanmean(np.min(np.abs(err), 0) ** 2)))
        stats[wid] = dict(A=A, t=t, per=per, med=med, orc=orc, oidx=oidx, n=n)
    return names, K, stats


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, names, K, stats):
        super().__init__()
        self.names, self.K, self.stats = names, K, stats
        self.setWindowTitle(f"v70 PF Viewer — {K}バンク × oracle  ({len(stats)}坑井)")
        self.resize(1400, 850)
        self.colors = [pg.mkColor(PALETTE[k % len(PALETTE)]) for k in range(K)]

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw); lay = QtWidgets.QHBoxLayout(cw)
        # --- 左: 検索 + 坑井リスト(硬い順) ---
        left = QtWidgets.QVBoxLayout(); lay.addLayout(left, 0)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText("坑井ID検索…"); self.search.textChanged.connect(self._filter)
        left.addWidget(self.search)
        self.sortbox = QtWidgets.QComboBox(); self.sortbox.addItems(["硬い順(median RMSE↓)", "oracle余地順(median−oracle↓)", "ID順"])
        self.sortbox.currentIndexChanged.connect(self._refill); left.addWidget(self.sortbox)
        self.lst = QtWidgets.QListWidget(); self.lst.setMinimumWidth(260); self.lst.currentItemChanged.connect(self._sel)
        left.addWidget(self.lst, 1)
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True); self.info.setTextFormat(QtCore.Qt.RichText)
        self.info.setFont(QtGui.QFont("Meiryo", 9)); left.addWidget(self.info)

        # --- 右: 上=軌跡, 下=oracleストリップ(x連動) ---
        glw = pg.GraphicsLayoutWidget(); lay.addWidget(glw, 1)
        self.p1 = glw.addPlot(row=0, col=0); self.p1.showGrid(x=True, y=True, alpha=0.2)
        self.p1.setLabel("left", "TVT (ft)"); self.p1.addLegend(offset=(10, 10), labelTextSize="8pt")
        glw.nextRow()
        self.p2 = glw.addPlot(row=1, col=0); self.p2.setMaximumHeight(120); self.p2.setXLink(self.p1)
        self.p2.setLabel("left", "oracleバンク"); self.p2.setLabel("bottom", "position (eval行)")
        self.p2.showGrid(y=True, alpha=0.2); self.p2.getAxis("left").setTicks([[(k, self.names[k].replace("pf_ancc_", "a").replace("pf_self_", "s").replace("pf_nbr_", "n")) for k in range(K)]])

        self._refill()
        if self.lst.count(): self.lst.setCurrentRow(0)

    def _order(self):
        m = self.sortbox.currentIndex()
        wids = list(self.stats)
        if m == 0: wids.sort(key=lambda w: -self.stats[w]["med"])
        elif m == 1: wids.sort(key=lambda w: -(self.stats[w]["med"] - self.stats[w]["orc"]))
        else: wids.sort()
        return wids

    def _refill(self):
        self.lst.blockSignals(True); self.lst.clear()
        f = self.search.text().strip().lower()
        for w in self._order():
            if f and f not in w.lower(): continue
            s = self.stats[w]
            it = QtWidgets.QListWidgetItem(f"{w[:14]}   med {s['med']:.1f} / orc {s['orc']:.1f}")
            it.setData(QtCore.Qt.UserRole, w); self.lst.addItem(it)
        self.lst.blockSignals(False)
        if self.lst.count(): self.lst.setCurrentRow(0)

    def _filter(self): self._refill()

    def _sel(self, cur, _prev=None):
        if cur is None: return
        self._plot(cur.data(QtCore.Qt.UserRole))

    def _plot(self, wid):
        s = self.stats[wid]; A, t, oidx, K = s["A"], s["t"], s["oidx"], self.K
        x = np.arange(s["n"])
        self.p1.clear(); self.p1.legend.clear() if self.p1.legend else None
        # 9本のPF(薄色)
        for k in range(K):
            self.p1.plot(x, A[k], pen=pg.mkPen(self.colors[k], width=1), name=self.names[k])
        # 真値(太線)
        self.p1.plot(x, t, pen=pg.mkPen("w", width=3), name="TRUE")
        # 真値上に oracleバンク色の点(=その行どのPFが最良か)
        brushes = [pg.mkBrush(self.colors[int(i)]) for i in oidx]
        self.p1.plot(x, t, pen=None, symbol="o", symbolSize=6, symbolBrush=brushes, symbolPen=None)
        self.p1.setTitle(f"{wid}   (点の色=その位置のoracle PF)")
        # 下: oracleバンク番号のステップ(色帯)
        self.p2.clear()
        self.p2.plot(x, oidx, pen=pg.mkPen("#888", width=1), stepMode=False)
        self.p2.plot(x, oidx, pen=None, symbol="s", symbolSize=5,
                     symbolBrush=[pg.mkBrush(self.colors[int(i)]) for i in oidx], symbolPen=None)
        self.p2.setYRange(-0.5, K - 0.5)
        # 統計パネル
        best = int(np.nanargmin(s["per"]))
        rows = "".join(f"<tr><td><font color='{PALETTE[k]}'>■</font> {self.names[k]}</td>"
                       f"<td align=right>{s['per'][k]:.2f}{' ★' if k == best else ''}</td>"
                       f"<td align=right>{100*np.mean(oidx == k):.0f}%</td></tr>" for k in range(K))
        self.info.setText(
            f"<b>{wid}</b>  ({s['n']}行)<br><table cellspacing=4>"
            f"<tr><td><b>bank</b></td><td align=right><b>RMSE</b></td><td align=right><b>oracle率</b></td></tr>"
            f"{rows}</table><br>"
            f"median = <b>{s['med']:.2f}</b><br>oracle = <b>{s['orc']:.2f}</b>  "
            f"(余地 {s['med']-s['orc']:.2f}ft)")


def main():
    if not PFP.exists():
        print(f"★{PFP.name} 無し → 先に `EVAL_STRIDE={STR} python v70/build_multipf_v70.py`"); return
    names, K, stats = load()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    v = Viewer(names, K, stats); v.show()
    if os.environ.get("SMOKE") == "1":       # 構築だけ確認して終了(GUI開かない)
        print(f"[viewer] OK: {K}バンク {len(stats)}坑井, ウィジェット構築成功"); return
    app.exec()


if __name__ == "__main__":
    main()
