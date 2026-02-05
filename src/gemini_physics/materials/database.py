"""
Material Database for Analogue-Gravity Optics (Phase 7.0+ Reconciliation).

Stores optical and physical properties of materials with wavelength-dependent
Sellmeier / Drude-Lorentz models.  All models return complex n (n + ik).

Canonical stack at 1550 nm (ChatGPT / Babar-Weaver reconciliation):
  Silicon  n = 3.476   (Li 1980)
  Gold     n = 0.19 + 10.35j  (Babar & Weaver 2015, eps = -107.13 + 3.93i)
  Sapphire n = 1.746   (Malitson 1962, ordinary ray)

Additional materials for overcoat / cladding studies:
  SiO2     n = 1.444   (Malitson 1965)
  Si3N4    n = 1.996   (Luke et al. 2015)
  Ta2O5    n = 2.058   (Gao et al. 2012)
  HfO2     n = 1.878   (Al-Kuhaili 2004)

Deprecated:
  Ice VIII -- replaced by Sapphire (comparable n~1.73, but stable at STP).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class PhaseOfMatter:
    name: str
    temperature_range_C: Tuple[float, float]
    pressure_range_Pa: Tuple[float, float]
    density_g_cm3: float


@dataclass
class Material:
    name: str
    density_g_cm3: float
    hardness_mohs: float
    phases: List[PhaseOfMatter] = field(default_factory=list)
    # Optical Data: lambda (nm) -> n, k
    _optical_data: Optional[Dict[str, np.ndarray]] = None  # {'wl': [], 'n': [], 'k': []}
    _optical_model: Optional[Callable[[float], complex]] = None

    def set_optical_data(self, wl_nm: np.ndarray, n: np.ndarray, k: np.ndarray = None):
        if k is None:
            k = np.zeros_like(n)
        self._optical_data = {'wl': wl_nm, 'n': n, 'k': k}
        self._n_interp = interp1d(wl_nm, n, kind='linear', fill_value="extrapolate")
        self._k_interp = interp1d(wl_nm, k, kind='linear', fill_value="extrapolate")

    def set_optical_model(self, model_func: Callable[[float], complex]):
        """Set a functional model (e.g. Sellmeier). Input: wl_nm, Output: complex n."""
        self._optical_model = model_func

    def refractive_index(self, wl_nm: float) -> complex:
        if self._optical_model:
            return self._optical_model(wl_nm)
        if self._optical_data:
            return self._n_interp(wl_nm) + 1j * self._k_interp(wl_nm)
        return 1.0 + 0j  # Default vacuum


class MaterialDatabase:
    def __init__(self):
        self.materials: Dict[str, Material] = {}
        self._initialize_defaults()

    def get(self, name: str) -> Material:
        default = Material(name="Vacuum", density_g_cm3=0.0, hardness_mohs=0.0)
        return self.materials.get(name, default)

    def _initialize_defaults(self):
        # --- Sapphire (Al2O3) ---
        # Malitson (1962) Sellmeier for Ordinary Ray.
        # Valid 0.2 -- 5.5 um.  n(1550nm) ~ 1.746.
        def sapphire_sellmeier(wl_nm):
            w = wl_nm / 1000.0  # microns
            ws = w ** 2
            n2 = (1 + 1.4313493 * ws / (ws - 0.0726631 ** 2)
                    + 0.65054713 * ws / (ws - 0.1193242 ** 2)
                    + 5.3414021 * ws / (ws - 18.028251 ** 2))
            return np.sqrt(n2) + 0j

        m_sapphire = Material("Sapphire (Al2O3)", 3.98, 9.0)
        m_sapphire.set_optical_model(sapphire_sellmeier)
        self.materials["Sapphire"] = m_sapphire

        # --- SiO2 (Fused Silica) ---
        # Malitson (1965) Sellmeier.  Valid 0.21 -- 3.71 um.
        # n(1550nm) ~ 1.4440.  Canonical Si-photonics cladding.
        def sio2_sellmeier(wl_nm):
            w = wl_nm / 1000.0
            ws = w ** 2
            n2 = (1 + 0.6961663 * ws / (ws - 0.0684043 ** 2)
                    + 0.4079426 * ws / (ws - 0.1162414 ** 2)
                    + 0.8974794 * ws / (ws - 9.896161 ** 2))
            return np.sqrt(n2) + 0j

        m_sio2 = Material("SiO2 (Fused Silica)", 2.20, 7.0)
        m_sio2.set_optical_model(sio2_sellmeier)
        self.materials["SiO2"] = m_sio2

        # --- Silicon (Si) ---
        # Salzberg & Villa (1957) Sellmeier, valid 1.36 -- 11 um.
        # n(1550nm) ~ 3.476.
        def silicon_sellmeier(wl_nm):
            w = wl_nm / 1000.0
            ws = w ** 2
            n2 = (1 + 10.6684293 * ws / (ws - 0.301516485 ** 2)
                    + 0.003043475 * ws / (ws - 1.13475115 ** 2)
                    + 1.54133408 * ws / (ws - 1104.0 ** 2))
            return np.sqrt(n2) + 0j

        m_si = Material("Silicon", 2.329, 7.0)
        m_si.set_optical_model(silicon_sellmeier)
        self.materials["Silicon"] = m_si

        # --- Gold (Au) ---
        # Tabulated from Babar & Weaver (2015), jhweaver.matse.illinois.edu.
        # Complex refractive index n + ik from measured eps1 + i*eps2.
        # n = sqrt((|eps| + eps1) / 2), k = sqrt((|eps| - eps1) / 2).
        # Interpolated for arbitrary wavelength in 500 -- 2000 nm range.
        _au_wl = np.array([500, 600, 700, 800, 1000, 1200, 1400, 1550, 1700, 2000],
                          dtype=np.float64)
        _au_n = np.array([0.86, 0.17, 0.13, 0.14, 0.16, 0.18, 0.18, 0.19, 0.21, 0.27],
                         dtype=np.float64)
        _au_k = np.array([1.84, 3.15, 4.13, 5.12, 6.92, 8.32, 9.54, 10.35, 11.05, 12.45],
                         dtype=np.float64)
        _au_n_interp = interp1d(_au_wl, _au_n, kind='cubic', fill_value='extrapolate')
        _au_k_interp = interp1d(_au_wl, _au_k, kind='cubic', fill_value='extrapolate')

        def gold_tabulated(wl_nm):
            return float(_au_n_interp(wl_nm)) + 1j * float(_au_k_interp(wl_nm))

        m_au = Material("Gold", 19.3, 2.5)
        m_au.set_optical_model(gold_tabulated)
        self.materials["Gold"] = m_au

        # --- Si3N4 (Stoichiometric Silicon Nitride) ---
        # Luke et al. (2015) Sellmeier.  Valid 0.31 -- 5.50 um.
        # n(1550nm) ~ 1.996.
        def si3n4_sellmeier(wl_nm):
            w = wl_nm / 1000.0
            ws = w ** 2
            n2 = (1 + 3.0249 * ws / (ws - 0.1353406 ** 2)
                    + 40314 * ws / (ws - 1239.842 ** 2))
            return np.sqrt(n2) + 0j

        m_sin = Material("Si3N4", 3.17, 8.5)
        m_sin.set_optical_model(si3n4_sellmeier)
        self.materials["Si3N4"] = m_sin

        # --- Ta2O5 (Tantalum Pentoxide) ---
        # Gao et al. (2012) Sellmeier.  Valid 0.3 -- 1.6 um.
        # n(1550nm) ~ 2.058.  Sweet spot between SiN and Si.
        def ta2o5_sellmeier(wl_nm):
            w = wl_nm / 1000.0
            ws = w ** 2
            # Gao et al. two-term Sellmeier
            n2 = (1 + 3.5554 * ws / (ws - 0.17834 ** 2)
                    + 0.29518 * ws / (ws - 15.0 ** 2))
            return np.sqrt(n2) + 0j

        m_ta2o5 = Material("Ta2O5", 8.20, 6.5)
        m_ta2o5.set_optical_model(ta2o5_sellmeier)
        self.materials["Ta2O5"] = m_ta2o5

        # --- HfO2 (Hafnia) ---
        # Al-Kuhaili (2004) Cauchy model.  n(1550nm) ~ 1.878.
        def hfo2_cauchy(wl_nm):
            w = wl_nm / 1000.0  # microns
            # Cauchy coefficients from Al-Kuhaili thin-film data
            n = 1.8503 + 0.0217 / w ** 2 + 0.0012 / w ** 4
            return n + 0j

        m_hfo2 = Material("HfO2", 9.68, 6.5)
        m_hfo2.set_optical_model(hfo2_cauchy)
        self.materials["HfO2"] = m_hfo2

        # --- Ice VIII (DEPRECATED) ---
        # Pan et al. (2014) 30 GPa -> n ~ 1.73.
        # Deprecated: requires 30 GPa, impractical for devices.
        # Use Sapphire (Al2O3, n~1.746) instead -- comparable n, STP-stable.
        def ice_viii_constant(wl_nm):
            return 1.73 + 0j  # constant proxy; real device should use Sapphire

        m_ice8 = Material("Ice VIII (30 GPa) [DEPRECATED]", 1.65, 6.0)
        m_ice8.set_optical_model(ice_viii_constant)
        self.materials["Ice_VIII"] = m_ice8
