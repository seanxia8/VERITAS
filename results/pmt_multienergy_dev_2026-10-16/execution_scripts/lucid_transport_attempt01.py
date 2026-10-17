"""Pinned LUCiD hard-intersection transport, before all PMT response.

The emitter supplies individual photons in metres/ns/nm, never SIREN weights.
Only first, forward, in-volume spherical sensor encounters are exported.
Absorption and dispersion are OFF for this bounded direct-light closure.
"""
from pathlib import Path
import hashlib
import numpy as np
from .photon_contract import PhotonBatch

PIN = "eaf2e89f579d6679e90020c732ac2c846895d691"


class PromptTransport:
    def __init__(self, config, manifest):
        import json, lucid
        upstream = Path(lucid.__file__).resolve().parent.parent
        expected = json.loads(Path(manifest).read_text())
        if expected["revision"] != PIN:
            raise ValueError("unreviewed upstream revision")
        for name, sha in expected["sha256"].items():
            if hashlib.sha256((upstream/name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"upstream source mismatch: {name}")
        import jax
        jax.config.update("jax_enable_x64", True)
        from lucid.geometry.detector_geometry import DetectorGeometry
        self.geometry = DetectorGeometry.from_config(str(config), temperature=None,
            deposit_leg_bound=True, max_candidates_per_ray=16)
        self.config_hash = hashlib.sha256(Path(config).read_bytes()).hexdigest()

    def export(self, event_id, photon_id, origins_m, directions, emission_ns, wavelength_nm):
        import jax.numpy as jnp
        o, d = np.asarray(origins_m,dtype=np.float64), np.asarray(directions,dtype=np.float64)
        ids, t, w = map(np.asarray, (photon_id, emission_ns, wavelength_nm))
        n = len(ids)
        if o.shape != (n, 3) or d.shape != (n, 3) or t.shape != (n,) or w.shape != (n,):
            raise ValueError("unaligned emission arrays")
        if not all(np.isfinite(x).all() for x in (o,d,t,w)) or not np.allclose(np.linalg.norm(d,axis=1),1,atol=2e-5):
            raise ValueError("finite emission and unit directions required")
        d=d/np.linalg.norm(d,axis=1,keepdims=True)
        if not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids)) != n or np.any(ids<0) or np.any(w<=0):
            raise ValueError("unique physical photon IDs and positive wavelengths required")
        if not np.asarray(self.geometry.detector.bounds_check(jnp.asarray(o))).all():
            raise ValueError("source photons must be inside the detector")
        if n == 0:
            return PhotonBatch(int(event_id), ids, ids.copy(), t, w).validate(self.geometry.num_sensors)
        # Fixed-size chunks prevent recompiling for each shower/rotation.
        chunks=[]
        for lo in range(0,n,4096):
            count=min(4096,n-lo)
            oo=np.zeros((4096,3),dtype=np.float64);dd=oo.copy();dd[:,2]=1
            oo[:count]=o[lo:lo+count];dd[:count]=d[lo:lo+count]
            r=self.geometry.propagator(jnp.asarray(oo),jnp.asarray(dd))
            chunks.append({k:np.asarray(r[k])[:,:count] for k in ('times','sensor_indices','inside_sensor')})
        result={k:np.concatenate([r[k] for r in chunks],axis=1) for k in chunks[0]}
        distance = np.asarray(result['times']).reshape(-1,n)
        sensor = np.asarray(result['sensor_indices']).reshape(-1,n)
        valid = np.asarray(result['inside_sensor']).reshape(-1,n) & (distance>0) & (sensor>=0)
        # Hard intersections, not rounded soft weights (upstream clips survival at 1-1e-6).
        first = np.argmin(np.where(valid,distance,np.inf),axis=0)
        keep = np.any(valid,axis=0)
        col = np.flatnonzero(keep); slot=first[keep]
        return PhotonBatch(int(event_id),ids[keep],sensor[slot,col].astype(np.int64),
            t[keep]+distance[slot,col]/float(self.geometry.speed_of_light),w[keep]).validate(self.geometry.num_sensors)

    def provenance(self):
        return dict(upstream_revision=PIN, geometry_config_sha256=self.config_hash,
            stage='incident_photons', applied_effects=[], prompt_direct=True,
            scattering=False, reflection=False, angular_acceptance='uniform',
            concentrator=False, absorption='off', dispersion='off',
            speed_m_per_ns=float(self.geometry.speed_of_light),
            transport_precision='float64; normalized directions', max_candidates_per_ray=16,
            sensor_model='upstream spherical hard intersections; first forward encounter',
            units=['m','ns','nm'], evidence_stage='development', non_citable=True)
