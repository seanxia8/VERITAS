# PMT calibration and protocol decisions requiring author input

This separates the authorized development validation from decisions that govern primary scientific claims. The multi-energy pilot uses primary starting vertex, initial direction and initial kinetic energy at three discrete muon energies; it does not freeze those as the paper's endpoints.

## Calibration inputs

| Decision/input | What is available | What the author or physics collaborator needs to supply or approve |
|---|---|---|
| Target hardware and response provenance | Pinned WCTE geometry; synthetic QE/CE, charge, T-Q and width functions | Confirm target PMT model(s), and either provide measured response data or explicitly approve a synthetic-response mechanism study with its limited claims |
| Photon detection | Wavelength preserved; QE/CE applied once | Absolute QE versus wavelength, CE convention/normalization, calibration uncertainty and valid wavelength support; do not import Super-K values without justification |
| Charge and timing | PE-scale gain, polynomial per-channel T-Q, charge-dependent Gaussian width | SPE gain/resolution, threshold convention, per-channel timing offsets and T-Q fits, valid PE intervals, and whether the Gaussian approximation is adequate or an asymmetric kernel is required |
| Physical groups | Real `mpmt_id` and `pmt_id_in_mpmt` exist in the pinned file and are aligned by its loader | Choose mPMT modules or electronics crates as the intervention groups; supply the crate/cable map if crates are intended. Module membership does not supply a magnetic-field response calibration |
| Geometry/medium fidelity | Upstream sensor spheres are snapped to a cylinder; prompt transport has no absorption, dispersion, scattering or reflections | The source is a water simulation cropped after generation to the detector volume, not a geometry-matched WCTE Geant4 run. Decide whether that geometry/transport approximation suffices for the intended claim and provide target optical calibration if it does not; specify any primary-only or prompt-time selection (the pilot preserves source emission times and secondary Cherenkov photons) |

## Protocol choices before primary evaluation

1. **Energy target and event support.** Choose initial particle energy on contained events, initial energy including escaping tracks, or a separately defined visible/deposited-energy endpoint. The present pilot contains only 3 fully contained placements out of 480, all at 500 MeV; a contained multi-energy study requires different event support or geometry. Escaping tracks do not deposit their full initial energy in WCTE. A charge fit cannot solve that loss of information by definition. The three-energy development pilot tests trainability, not continuous-energy interpolation or an adopted harm endpoint.
2. **Track target.** Confirm whether the consequence endpoint is the primary starting vertex, direction, track axis, stopping point, or a specified combination. Starting vertex and direction are the development targets; the point-source fitter is not used on tracks.
3. **D-K and failure policy.** Choose endpoint aggregation and whether a failed endpoint makes a cell unevaluable. Retain endpoint errors and failure rates; no silent complete-case selection. Confirm the clean denominator policy and handling of zero denominators.
4. **Threshold and dead-map policy.** Approve the primary harm threshold (the old 1.10 is still provisional) and whether corrected or uncorrected energy is primary. Calibration must remain disjoint from model fitting and held-out evaluation.
5. **S families and final holdouts.** Confirm allowed physical position/direction/energy support, family/group/severity/seed holdouts, the continuous-energy data needed, and precision-driven sample sizes. Development test events are not an untouched confirmation set.
6. **Primary model and comparison.** Confirm model outputs/hooks and the same-operational-information baseline comparison. The small PointNet is an engineering subject, not an automatic replacement for the preregistered primary architecture or `GENERIC_COMMITTED`.

The highest-priority next author decision is the energy target and containment policy. That determines which data must be generated before a scientifically interpretable energy-harm campaign. Hardware calibration can be collected in parallel. No decision above is inferred from a favorable development result.
