"""Parameter spaces: bounds, nominal values, log scaling and the unit-cube map."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Param:
    key: str
    low: float
    high: float
    nominal: float | None = None
    log: bool = False
    name: str = ""
    unit: str = ""
    cls: str = ""

    def __post_init__(self):
        if self.nominal is None:
            self.nominal = 0.5 * (self.low + self.high)
        if self.log and (self.low <= 0 or self.high <= 0):
            raise ValueError(f"log-scaled parameter {self.key} needs positive bounds")
        self.name = self.name or self.key


@dataclass
class Space:
    params: list[Param] = field(default_factory=list)

    @property
    def keys(self):
        return [p.key for p in self.params]

    @property
    def dim(self):
        return len(self.params)

    def __getitem__(self, k):
        return self.params[self.keys.index(k)] if isinstance(k, str) else self.params[k]

    def subset(self, keys):
        return Space([self[k] for k in keys])

    def _tr(self):
        lo = np.array([np.log10(p.low) if p.log else p.low for p in self.params])
        hi = np.array([np.log10(p.high) if p.log else p.high for p in self.params])
        lg = np.array([p.log for p in self.params])
        return lo, hi, lg

    def to_unit(self, X):
        X = np.atleast_2d(np.asarray(X, float)).copy()
        lo, hi, lg = self._tr()
        X[:, lg] = np.log10(np.clip(X[:, lg], 1e-300, None))
        return (X - lo) / np.where(hi > lo, hi - lo, 1.0)

    def from_unit(self, U):
        U = np.atleast_2d(np.asarray(U, float))
        lo, hi, lg = self._tr()
        X = lo + U * (hi - lo)
        X[:, lg] = 10 ** X[:, lg]
        return X

    @property
    def nominal(self):
        return np.array([p.nominal for p in self.params], float)

    def outside(self, X, tol=1e-9):
        """Boolean mask (n, d) of values outside the declared bounds (extrapolation)."""
        U = self.to_unit(X)
        return (U < -tol) | (U > 1 + tol)

    @classmethod
    def from_records(cls, recs):
        return cls([Param(**{k: v for k, v in r.items() if k in Param.__dataclass_fields__}) for r in recs])

    @classmethod
    def from_data(cls, df, keys):
        return cls([Param(k, float(df[k].min()), float(df[k].max())) for k in keys])
