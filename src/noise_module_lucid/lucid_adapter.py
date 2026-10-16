"""Strict pre-response interception adapter for LUCiD eaf2e89.

The ordinary simulator return is NOT this record. An upstream interception
must export incident sensor encounters before sensor_response QE/TTS and
retain original photon identity. Wavelength mode alone is insufficient.
"""
from .photon_contract import PhotonBatch

PINNED_REVISION = "eaf2e89"


def incident_record(record, *, n_channels):
    required = {"event_id", "photon_id", "sensor_id", "arrival_time_ns", "wavelength_nm",
                "stage", "applied_effects", "schema", "units", "upstream_revision",
                "prompt_direct", "scattering", "reflection", "angular_acceptance",
                "concentrator", "absorption", "dispersion"}
    if required - record.keys():
        raise ValueError(f"missing adapter fields: {sorted(required-record.keys())}")
    if record["upstream_revision"] != PINNED_REVISION:
        raise ValueError("unreviewed upstream revision")
    if record["stage"] != "incident_photons" or record["applied_effects"]:
        raise ValueError("adapter needs pre-QE/pre-TTS incident photons")
    if record["prompt_direct"] is not True or record["scattering"] is not False or record["reflection"] is not False or record["concentrator"] is not False or record["angular_acceptance"] != "uniform":
        raise ValueError("first trial requires prompt direct transport and uniform acceptance")
    if record["absorption"] not in ("off", "upstream") or record["dispersion"] not in ("off", "upstream"):
        raise ValueError("transport ownership must be declared")
    return PhotonBatch(**{k: record[k] for k in PhotonBatch.__dataclass_fields__}).validate(n_channels)
