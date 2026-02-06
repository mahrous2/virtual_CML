from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import uuid

import numpy as np
import pandas as pd

# Same legacy 8-channel API your UI expects (ch410..ch670).
# This is a browser-friendly version: no Flask, no colour-science. (Your original backend used Flask.)  # [page:45]


@dataclass
class ColorMixingResult:
    redvolume: float
    yellowvolume: float
    bluevolume: float
    wellposition: str
    sensordata: Dict[str, int]
    experimentid: str
    timestamp: str


class VirtualColorMixingLab:
    """
    High-fidelity software simulation (Beer-Lambert + AS7341-like internal model),
    preserving the legacy 8-channel API used by your current UI.
    """

    def __init__(
        self,
        platerows: int = 8,
        platecols: int = 12,
        randomseed: Optional[int] = None,
        systematicnoiselevel: float = 0.05,
        randomnoiselevel: float = 0.01,
    ):
        self.platerows = int(platerows)
        self.platecols = int(platecols)
        self.systematicnoiselevel = float(systematicnoiselevel)
        self.randomnoiselevel = float(randomnoiselevel)
        self.rng = np.random.default_rng(randomseed)

        self.nativechannels = ["ch415", "ch445", "ch480", "ch515", "ch555", "ch590", "ch630", "ch680", "clear"]
        self.legacychannels = ["ch410", "ch440", "ch470", "ch510", "ch550", "ch583", "ch620", "ch670"]
        self.legacytonative = {
            "ch410": "ch415",
            "ch440": "ch445",
            "ch470": "ch480",
            "ch510": "ch515",
            "ch550": "ch555",
            "ch583": "ch590",
            "ch620": "ch630",
            "ch670": "ch680",
        }

        # Tuned absorption coefficients (dimensionless)
        self.absorptions = {
            "red":  {"ch415": 1.8, "ch445": 1.5, "ch480": 1.2, "ch515": 0.8, "ch555": 0.3, "ch590": 0.1, "ch630": 0.02, "ch680": 0.01, "clear": 0.5},
            "yel":  {"ch415": 2.8, "ch445": 2.2, "ch480": 0.8, "ch515": 0.2, "ch555": 0.05,"ch590": 0.02,"ch630": 0.01,"ch680": 0.00,"clear": 0.4},
            "blu":  {"ch415": 0.05,"ch445": 0.1, "ch480": 0.4, "ch515": 1.1, "ch555": 2.2, "ch590": 2.9, "ch630": 3.5, "ch680": 3.2, "clear": 0.9},
        }

        # Baseline intensity I0 for each native channel
        self.baselinei0 = {"ch415": 1200, "ch445": 1800, "ch480": 2500, "ch515": 1000, "ch555": 3800, "ch590": 4200, "ch630": 4800, "ch680": 4500, "clear": 6200}

        self.resultslog: List[ColorMixingResult] = []
        self.experimentcounter = 0
        self._generate_systematic_well_map()

    def _rowcol_to_well(self, row: int, col: int) -> str:
        return f"{chr(65 + int(row))}{int(col) + 1}"

    def _well_to_rowcol(self, well: str) -> Tuple[int, int]:
        well = well.strip()
        row = ord(well[0].upper()) - 65
        col = int(well[1:]) - 1
        return row, col

    def _validate_volumes(self, r: float, y: float, b: float) -> Optional[str]:
        if not all(1 <= float(v) <= 300 for v in (r, y, b)):
            return "Each volume must be between 1 and 300 µL"
        total = float(r) + float(y) + float(b)
        if total > 300:
            return f"Total volume {total} µL exceeds 300 µL limit"
        return None

    def _generate_systematic_well_map(self) -> None:
        rowlin = np.linspace(-1, 1, self.platerows)
        collin = np.linspace(-1, 1, self.platecols)
        rr, cc = np.meshgrid(rowlin, collin, indexing="ij")
        dist = np.sqrt(rr**2 + cc**2)
        dist = dist / dist.max() if dist.max() else 1.0
        field = self.rng.normal(0.0, 1.0, size=(self.platerows, self.platecols))
        wf = 1.0 + self.systematicnoiselevel * dist + self.systematicnoiselevel * 0.2 * field
        self.wellsystematicfactor = wf / wf.mean()

    def _pipetting_error(self, target: float) -> float:
        target = float(target)
        if target < 10:
            cv = 0.12
        elif target < 50:
            cv = 0.04
        else:
            cv = 0.015
        actual = target + self.rng.normal(0.0, cv * target)
        return float(max(0.1, actual))

    def _native_sensor_readings(self, rvol: float, yvol: float, bvol: float, well: str) -> Dict[str, int]:
        totalvol = float(rvol + yvol + bvol)
        if totalvol <= 0:
            return {ch: 0 for ch in self.nativechannels}

        cred = float(rvol) / totalvol
        cyel = float(yvol) / totalvol
        cblu = float(bvol) / totalvol

        pathlength = totalvol / 300.0
        row, col = self._well_to_rowcol(well)
        sysfactor = float(self.wellsystematicfactor[row, col])

        out: Dict[str, int] = {}
        for ch in self.nativechannels:
            i0 = float(self.baselinei0[ch])
            od = (
                self.absorptions["red"][ch] * cred
                + self.absorptions["yel"][ch] * cyel
                + self.absorptions["blu"][ch] * cblu
            )
            val = i0 * float(np.exp(-od * pathlength))
            noisy = val * sysfactor * float(self.rng.normal(1.0, self.randomnoiselevel))
            out[ch] = int(np.clip(noisy, 0, 65535))
        return out

    def _calculate_sensor_readings(self, rvol: float, yvol: float, bvol: float, well: str) -> Dict[str, int]:
        native = self._native_sensor_readings(rvol, yvol, bvol, well)
        legacy = {legacych: int(native[nativech]) for legacych, nativech in self.legacytonative.items()}
        return legacy

    def mix_colors(self, R: float, Y: float, B: float, wellposition: Optional[str] = None) -> ColorMixingResult:
        msg = self._validate_volumes(R, Y, B)
        if msg:
            raise ValueError(msg)

        if wellposition is None:
            row = (self.experimentcounter // self.platecols) % self.platerows
            col = self.experimentcounter % self.platecols
            wellposition = self._rowcol_to_well(row, col)

        ractual = self._pipetting_error(R)
        yactual = self._pipetting_error(Y)
        bactual = self._pipetting_error(B)

        sensordata = self._calculate_sensor_readings(ractual, yactual, bactual, wellposition)

        res = ColorMixingResult(
          redvolume=float(R),
          yellowvolume=float(Y),
          bluevolume=float(B),
          wellposition=str(wellposition),
          sensordata=sensordata,
          experimentid=str(uuid.uuid4()),
          timestamp=pd.Timestamp.now().isoformat(),
        )
        self.resultslog.append(res)
        self.experimentcounter += 1
        return res

    def run_experiment_batch(self, experimentdesign: List[Dict]) -> pd.DataFrame:
        rows = []
        for exp in experimentdesign:
            r = exp.get("R", exp.get("Red", exp.get("red")))
            y = exp.get("Y", exp.get("Yellow", exp.get("yellow")))
            b = exp.get("B", exp.get("Blue", exp.get("blue")))
            well = exp.get("well", exp.get("Well"))
            res = self.mix_colors(r, y, b, wellposition=well)
            rows.append({
                "Red": res.redvolume,
                "Yellow": res.yellowvolume,
                "Blue": res.bluevolume,
                "well": res.wellposition,
                **res.sensordata,
            })
        return pd.DataFrame(rows)

    def export_results_df(self) -> pd.DataFrame:
        out = []
        for r in self.resultslog:
            out.append({
                "Red": r.redvolume,
                "Yellow": r.yellowvolume,
                "Blue": r.bluevolume,
                "well": r.wellposition,
                **r.sensordata,
                "timestamp": r.timestamp,
                "experimentid": r.experimentid,
            })
        return pd.DataFrame(out)
