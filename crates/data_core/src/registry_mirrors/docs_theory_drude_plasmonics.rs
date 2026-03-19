//! # Drude Material Regimes and Refractory Plasmonics
//!
//! ## Overview
//!
//! The Drude model of electrical conduction provides a foundational phenomenological framework for describing the optical and transport properties of materials with mobile charge carriers. However, there is no finite, universally accepted "complete list" of Drude materials. "Drude-ness" is not a membership label; rather, it is a *regime* wherein a material's free carriers (electrons, holes, or ions) behave as a damped charge fluid characterized by a single momentum-relaxation time ($\tau$). 
//!
//! Consequently, any material possessing mobile charge carriers can exhibit a Drude response within a specific frequency or temperature window where the free-carrier term dominates its complex permittivity or conductivity.
//!
//! ### Core Drude Model
//!
//! In the frequency domain (angular frequency $\omega$), the complex relative permittivity is typically expressed as:
//!
//! $$ \varepsilon(\omega) = \varepsilon_\infty - \frac{\omega_p^2}{\omega^2 + i \gamma \omega} $$
//!
//! Where:
//! * $\varepsilon_\infty$: Background permittivity from higher-energy bound-electron polarization.
//! * $\omega_p$: Unscreened plasma frequency ($\omega_p^2 = \frac{n e^2}{\varepsilon_0 m^*}$).
//! * $\gamma = 1/\tau$: Scattering rate (damping).
//!
//! ## Elemental Metals
//!
//! Most elemental metals exhibit a low-frequency Drude-like intraband response. However, interband transitions, multiple carrier pockets, and strong correlations often necessitate a transition to a **Drude-Lorentz** or **Extended Drude** formulation, particularly in the visible and UV spectrum. 
//!
//! The Gororoba framework incorporates the canonical 11-metal Drude-Lorentz parameter set (e.g., from Rakic et al.) which provides the Drude oscillator strength ($f_0$) and damping ($\Gamma_0$), yielding an effective Drude plasma energy $\omega_{p,\text{eff}} = \omega_p \sqrt{f_0}$. 
//!
//! Included metals: Silver (Ag), Gold (Au), Copper (Cu), Aluminum (Al), Beryllium (Be), Chromium (Cr), Nickel (Ni), Palladium (Pd), Platinum (Pt), Titanium (Ti), and Tungsten (W). 
//!
//! ## Non-Elemental Drude Material Families
//!
//! To extend plasmonic devices beyond the limitations of noble metals (e.g., high losses in the visible, CMOS incompatibility, and low thermal stability), several key material families have been established:
//!
//! ### 1. Transition Metal Nitrides (Refractory Plasmonics)
//! These materials offer high-temperature stability, mechanical hardness, and CMOS-compatibility. Their localized surface plasmon resonance (LSPR) bands are tunable via stoichiometry.
//! * **Titanium Nitride (TiN):** An established alternative to Gold. It exhibits an unscreened plasma energy ranging from 4.5 to 9.8 eV depending on the deposition method (e.g., sputtering vs. epitaxy). Screened plasma energy crosses zero in the UV/visible ($\sim 2.5$ eV).
//! * **Zirconium Nitride (ZrN):** Provides slightly lower losses than TiN in the visible region. $\omega_p \approx 7.456$ eV.
//! * **Hafnium Nitride (HfN):** Evaluated for bulk plasmons around 370 nm. $\omega_p \approx 8.19$ eV.
//!
//! ### 2. Transparent Conducting Oxides (TCOs)
//! TCOs are heavily doped semiconductors where the epsilon-near-zero (ENZ) crossover can be tuned deeply into the Near-Infrared (NIR) and telecom bands. To behave as "metals" in the NIR, carrier densities typically must exceed $10^{20} \text{ cm}^{-3}$.
//! * **Indium Tin Oxide (ITO):** Widely utilized; adjustable carrier density yields effective mass $m^* \approx 0.4 m_0$.
//! * **Aluminum-doped Zinc Oxide (AZO) & Gallium-doped Zinc Oxide (GZO):** Cross-over wavelengths can fall below the telecom 1.55 $\mu$m band.
//!
//! ### 3. Disordered Conductors (Drude-Smith)
//! For nanoporous metals, ultrathin films, and conducting polymers (e.g., PEDOT:PSS), strong backscattering and localization phenomena break the simple Drude approximation. The **Drude-Smith** model introduces a persistence of velocity parameter ($c \in [-1, 0]$), where $c < 0$ denotes suppression of DC conductivity due to carrier localization.
//!
//! ## Implementation in Gororoba Materials Core
//!
//! The optical responses of these generalized Drude regimes have been systematically integrated into the `materials_core` crate:
//! * Refractory nitrides (`TiN`, `ZrN`, `HfN`) have been added with `DrudeLorentzParams` matched to robust literature compilations.
//! * TCOs (`AZO`, `ITO`) are modeled utilizing `ExtendedDrude` components.
//! * Disordered conductors (`PEDOT:PSS`) explicitly utilize the `ScatteringModel::DrudeSmith` capability, enforcing frequency-dependent scattering that captures localized backscattering geometries.
//!
//! ## References
//! 1. Patsalas, P., et al. (2015). *Optical Properties and Plasmonic Performance of Titanium Nitride*. Materials.
//! 2. Naik, G. V., et al. (2013). *Alternative Plasmonic Materials: Beyond Gold and Silver*. Advanced Materials.
//! 3. Rakic, A. D., et al. (1998). *Optical properties of metallic films for vertical-cavity optoelectronic devices*. Applied Optics.
//!
