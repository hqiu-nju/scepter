from __future__ import annotations

import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

from scepter import satsim
import scepter.nbeam as nbeam
from scepter.nbeam import BeamCapSizingConfig, run_beam_cap_sizing


REPO_ROOT = Path(__file__).resolve().parents[2]


class _SliceRecordingDataset:
    def __init__(self, data, *, chunks=None, compression=None, shuffle=False):
        arr = np.asarray(data)
        self._data = arr
        self.shape = arr.shape
        self.ndim = arr.ndim
        self.dtype = arr.dtype
        self.chunks = chunks
        self.compression = compression
        self.shuffle = shuffle
        self.requests: list[tuple[int, int]] = []

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = 0 if item.start is None else int(item.start)
            stop = int(self.shape[0]) if item.stop is None else int(item.stop)
            self.requests.append((start, stop))
        return self._data[item]


class _LayoutOnlyDataset:
    def __init__(self, *, shape, dtype, chunks=None, compression=None, shuffle=False):
        self.shape = tuple(int(v) for v in shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(dtype)
        self.chunks = chunks
        self.compression = compression
        self.shuffle = shuffle


def _write_iter_group(
    path,
    *,
    eligible_mask=None,
    eligible_csr=None,
    counts=None,
    count_name="sat_beam_counts_eligible",
    belt=None,
    az=None,
    el=None,
    times=None,
) -> None:
    with h5py.File(path, "w") as h5:
        g_iter = h5.create_group("iter")
        g0 = g_iter.create_group("iter_00000")
        if eligible_mask is not None:
            g0.create_dataset("sat_eligible_mask", data=np.asarray(eligible_mask, dtype=np.bool_))
        if eligible_csr is not None:
            row_ptr_ds = g0.create_dataset(
                "sat_eligible_csr_row_ptr",
                data=np.asarray(eligible_csr["sat_eligible_csr_row_ptr"], dtype=np.int64),
            )
            g0.create_dataset(
                "sat_eligible_csr_sat_idx",
                data=np.asarray(eligible_csr["sat_eligible_csr_sat_idx"], dtype=np.int32),
            )
            row_ptr_ds.attrs["sat_eligible_csr_time_count"] = int(eligible_csr["sat_eligible_csr_time_count"])
            row_ptr_ds.attrs["sat_eligible_csr_cell_count"] = int(eligible_csr["sat_eligible_csr_cell_count"])
            row_ptr_ds.attrs["sat_eligible_csr_sat_count"] = int(eligible_csr["sat_eligible_csr_sat_count"])
        if counts is not None:
            g0.create_dataset(str(count_name), data=np.asarray(counts, dtype=np.int32))
        if belt is not None:
            g0.create_dataset("sat_belt_id", data=np.asarray(belt, dtype=np.int16))
        if az is not None:
            g0.create_dataset("sat_azimuth", data=np.asarray(az, dtype=np.float32))
        if el is not None:
            g0.create_dataset("sat_elevation", data=np.asarray(el, dtype=np.float32))
        if times is not None:
            g0.create_dataset("times", data=np.asarray(times, dtype=np.float64))


def test_run_beam_cap_sizing_supports_pure_reroute_only():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "pure_only.h5"
        eligible_mask = np.array(
            [
                [
                    [True, True],
                    [True, False],
                ],
                [
                    [True, True],
                    [False, True],
                ],
            ],
            dtype=bool,
        )
        _write_iter_group(storage, eligible_mask=eligible_mask)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert result["selected_caps"]["pure_reroute"] == 1
        curve = result["policy_curves"]["pure_reroute"]
        np.testing.assert_allclose(curve["delta"][:2], np.array([1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(curve["eps"][:2], np.array([1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)
        assert result["output_run_dir"] is None
        assert result["fig_tail"] is None
        assert result["fig_delta"] is None
        assert result["fig_eps"] is None


def test_run_beam_cap_sizing_rejects_removed_global_policy_alias():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "removed_global_alias.h5"
        _write_iter_group(storage, counts=np.array([[1]], dtype=np.int32))

        with pytest.raises(ValueError, match="Unknown policy keys.*global"):
            run_beam_cap_sizing(
                storage,
                config=BeamCapSizingConfig(
                    enabled_policy_keys=("global",),
                    nco=1,
                    save_outputs=False,
                    show_plots=False,
                    enable_skycell_demand_vis=False,
                ),
            )


def test_run_beam_cap_sizing_accepts_sat_beam_counts_used():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "counts_used.h5"
        _write_iter_group(
            storage,
            counts=np.array([[1]], dtype=np.int32),
            count_name="sat_beam_counts_used",
        )

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("no_reroute",),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert result["selected_caps"]["no_reroute"] == 1


def test_run_beam_cap_sizing_accepts_boresight_sky_resolved_counts():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "boresight_counts_3d.h5"
        counts = np.asarray(
            [
                [[1, 0], [0, 2]],
                [[0, 1], [3, 0]],
            ],
            dtype=np.int32,
        )
        _write_iter_group(
            storage,
            counts=counts,
            count_name="sat_beam_counts_used",
        )

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("no_reroute",),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert "no_reroute" in result["policy_curves"]
        assert result["selected_caps"]["no_reroute"] is not None


def test_run_beam_cap_sizing_accepts_stored_boresight_counts_shape():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "boresight_counts_4d.h5"
        counts_sky = np.asarray(
            [
                [[1, 0], [0, 2]],
                [[0, 1], [3, 0]],
            ],
            dtype=np.int32,
        )
        counts = np.transpose(counts_sky, (0, 2, 1))[:, None, :, :]
        _write_iter_group(
            storage,
            counts=counts,
            count_name="sat_beam_counts_used",
        )

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("no_reroute",),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert "no_reroute" in result["policy_curves"]
        assert result["selected_caps"]["no_reroute"] is not None


def test_run_beam_cap_sizing_supports_boresight_dense_pure_reroute():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "boresight_pure_dense.h5"
        eligible_mask = np.asarray(
            [
                [
                    [[True, False]],
                    [[False, True]],
                ],
                [
                    [[True, True]],
                    [[False, False]],
                ],
            ],
            dtype=bool,
        )
        _write_iter_group(storage, eligible_mask=eligible_mask)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert "pure_reroute" in result["policy_curves"]
        assert result["selected_caps"]["pure_reroute"] is not None


def test_run_beam_cap_sizing_rejects_unsupported_count_shape():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "unsupported_counts.h5"
        _write_iter_group(
            storage,
            counts=np.zeros((1, 1, 1, 1, 1), dtype=np.int32),
            count_name="sat_beam_counts_used",
        )

        with pytest.raises(ValueError, match="Supported layouts are"):
            run_beam_cap_sizing(
                storage,
                config=BeamCapSizingConfig(
                    enabled_policy_keys=("no_reroute",),
                    nco=1,
                    save_outputs=False,
                    show_plots=False,
                    enable_skycell_demand_vis=False,
                ),
            )


def test_run_beam_cap_sizing_can_run_quietly(monkeypatch, capsys):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "quiet_counts.h5"
        _write_iter_group(
            storage,
            counts=np.asarray([[1, 2], [0, 1]], dtype=np.int32),
            count_name="sat_beam_counts_used",
        )

        monkeypatch.setattr(nbeam, "_load_tqdm", lambda: None)

        run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("no_reroute",),
                nco=1,
                emit_progress_output=False,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


def test_run_beam_cap_sizing_explains_beam_demand_count_is_not_count_source():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "beam_demand_only.h5"
        with h5py.File(storage, "w") as h5:
            g_iter = h5.create_group("iter")
            g0 = g_iter.create_group("iter_00000")
            g0.create_dataset("beam_demand_count", data=np.asarray([1, 2, 3], dtype=np.int32))

        with pytest.raises(KeyError, match="sat_beam_counts_used"):
            run_beam_cap_sizing(
                storage,
                config=BeamCapSizingConfig(
                    enabled_policy_keys=("no_reroute",),
                    nco=1,
                    save_outputs=False,
                    show_plots=False,
                    enable_skycell_demand_vis=False,
                ),
            )


def test_run_beam_cap_sizing_supports_pure_reroute_csr_only():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "pure_only_csr.h5"
        eligible_mask = np.array(
            [
                [
                    [True, True],
                    [True, False],
                ],
                [
                    [True, True],
                    [False, True],
                ],
            ],
            dtype=bool,
        )
        eligible_csr = satsim._pure_reroute_dense_mask_to_csr_payload(eligible_mask)
        _write_iter_group(storage, eligible_csr=eligible_csr)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert result["selected_caps"]["pure_reroute"] == 1
        curve = result["policy_curves"]["pure_reroute"]
        np.testing.assert_allclose(curve["delta"][:2], np.array([1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(curve["eps"][:2], np.array([1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)


def test_run_beam_cap_sizing_prefers_csr_when_dense_and_sparse_are_both_present(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "pure_both.h5"
        eligible_mask = np.array(
            [
                [
                    [True, True, False],
                    [True, False, True],
                ]
            ],
            dtype=bool,
        )
        eligible_csr = satsim._pure_reroute_dense_mask_to_csr_payload(eligible_mask)
        _write_iter_group(storage, eligible_mask=eligible_mask, eligible_csr=eligible_csr)

        seen: dict[str, bool] = {"csr": False}

        def _wrapped_cpu_solver(eligible_chunk, *, nco, beam_caps):
            seen["csr"] = isinstance(eligible_chunk, dict)
            return satsim.pure_reroute_service_curve(eligible_chunk, nco=nco, beam_caps=beam_caps)

        monkeypatch.setattr(nbeam, "_run_pure_reroute_cpu_solver", _wrapped_cpu_solver)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                nco=1,
                pure_reroute_backend="cpu",
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert seen["csr"] is True
        assert result["selected_caps"]["pure_reroute"] == 1


def test_run_beam_cap_sizing_supports_mixed_policies():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "mixed.h5"
        eligible_mask = np.array(
            [
                [
                    [True, True, False],
                    [True, False, True],
                ]
            ],
            dtype=bool,
        )
        counts = np.array([[1, 1, 0]], dtype=np.int32)
        _write_iter_group(storage, eligible_mask=eligible_mask, counts=counts)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute", "pure_reroute"),
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert set(result["selected_caps"]) == {"full_reroute", "pure_reroute"}
        assert result["selected_caps"]["full_reroute"] >= 1
        assert result["selected_caps"]["pure_reroute"] >= 1


def test_run_beam_cap_sizing_supports_mixed_dense_and_csr_iterations():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "mixed_dense_csr_iters.h5"
        dense_mask = np.array(
            [
                [
                    [True, True],
                    [True, False],
                ]
            ],
            dtype=bool,
        )
        csr_mask = np.array(
            [
                [
                    [True, False],
                    [False, True],
                ]
            ],
            dtype=bool,
        )
        csr_payload = satsim._pure_reroute_dense_mask_to_csr_payload(csr_mask)
        with h5py.File(storage, "w") as h5:
            g_iter = h5.create_group("iter")
            g0 = g_iter.create_group("iter_00000")
            g0.create_dataset("sat_eligible_mask", data=dense_mask.astype(np.bool_))
            g1 = g_iter.create_group("iter_00001")
            row_ptr_ds = g1.create_dataset(
                "sat_eligible_csr_row_ptr",
                data=np.asarray(csr_payload["sat_eligible_csr_row_ptr"], dtype=np.int64),
            )
            g1.create_dataset(
                "sat_eligible_csr_sat_idx",
                data=np.asarray(csr_payload["sat_eligible_csr_sat_idx"], dtype=np.int32),
            )
            row_ptr_ds.attrs["sat_eligible_csr_time_count"] = int(csr_payload["sat_eligible_csr_time_count"])
            row_ptr_ds.attrs["sat_eligible_csr_cell_count"] = int(csr_payload["sat_eligible_csr_cell_count"])
            row_ptr_ds.attrs["sat_eligible_csr_sat_count"] = int(csr_payload["sat_eligible_csr_sat_count"])

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                nco=1,
                pure_reroute_backend="cpu",
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        curve = result["policy_curves"]["pure_reroute"]
        assert curve["delta"].shape[0] >= 2


def test_run_beam_cap_sizing_requires_eligible_mask_for_pure_reroute():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "missing_mask.h5"
        _write_iter_group(storage, counts=np.array([[1, 0, 1]], dtype=np.int32))

        with pytest.raises(KeyError, match="sat_eligible_mask"):
            run_beam_cap_sizing(
                storage,
                config=BeamCapSizingConfig(
                    enabled_policy_keys=("pure_reroute",),
                    nco=1,
                    save_outputs=False,
                    show_plots=False,
                    enable_skycell_demand_vis=False,
                ),
            )


def test_run_beam_cap_sizing_respects_output_dir_override():
    with TemporaryDirectory(dir=".") as tmp_dir:
        tmp_path = Path(tmp_dir)
        storage = tmp_path / "output_dir.h5"
        eligible_mask = np.array([[[True, True], [True, False]]], dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)
        out_root = tmp_path / "custom_output_root"

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                nco=1,
                save_outputs=True,
                output_dir=out_root,
                output_prefix="beamcaps",
                save_plots_html=False,
                save_results_json=False,
                save_results_npz=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        run_dir = result["output_run_dir"]
        assert run_dir is not None
        run_dir_path = Path(run_dir)
        assert out_root.exists()
        assert run_dir_path.exists()
        assert run_dir_path.parent == out_root


def test_select_auto_pure_reroute_backend_prefers_cpu_when_faster(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "auto_cpu.h5"
        eligible_mask = np.array([[[True, True], [True, False]]], dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)

        cpu_calls = {"count": 0}
        gpu_calls = {"count": 0}

        class _FakeGpuSession:
            def __init__(self, *args, **kwargs):
                return None

            def close(self, reset_device=False):
                return None

        monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: True)
        monkeypatch.setattr(
            nbeam,
            "_collect_pure_reroute_probe_buffer",
            lambda *args, **kwargs: (eligible_mask.astype(bool, copy=False), 1, int(np.count_nonzero(eligible_mask))),
        )
        monkeypatch.setattr(nbeam, "_run_pure_reroute_cpu_solver", lambda *args, **kwargs: cpu_calls.__setitem__("count", cpu_calls["count"] + 1) or {})
        monkeypatch.setattr(nbeam, "_run_pure_reroute_gpu_solver", lambda *args, **kwargs: gpu_calls.__setitem__("count", gpu_calls["count"] + 1) or {})

        import scepter.gpu_accel as gpu_accel
        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _FakeGpuSession)
        perf_values = itertools.chain((0.0, 1.0, 1.0, 2.4), itertools.repeat(2.4))
        monkeypatch.setattr(nbeam.time, "perf_counter", lambda: next(perf_values))

        with h5py.File(storage, "r") as h5:
            diag = nbeam._select_auto_pure_reroute_backend(
                h5,
                ["iter_00000"],
                count_var=None,
                nco=1,
                beam_caps=np.arange(2, dtype=np.int32),
                max_demand_slots=None,
            )

        assert diag["pure_reroute_backend_selected"] == "cpu"
        assert cpu_calls["count"] == 2
        assert gpu_calls["count"] == 2
        assert diag["pure_reroute_probe_cpu_s"] == pytest.approx(1.0)
        assert diag["pure_reroute_probe_gpu_s"] == pytest.approx(1.4)


def test_select_auto_pure_reroute_backend_prefers_gpu_when_enough_faster(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "auto_gpu_probe.h5"
        eligible_mask = np.array([[[True, True], [True, False]]], dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)

        class _FakeGpuSession:
            def __init__(self, *args, **kwargs):
                return None

            def close(self, reset_device=False):
                return None

        monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: True)
        monkeypatch.setattr(
            nbeam,
            "_collect_pure_reroute_probe_buffer",
            lambda *args, **kwargs: (eligible_mask.astype(bool, copy=False), 1, int(np.count_nonzero(eligible_mask))),
        )
        monkeypatch.setattr(nbeam, "_run_pure_reroute_cpu_solver", lambda *args, **kwargs: {})
        monkeypatch.setattr(nbeam, "_run_pure_reroute_gpu_solver", lambda *args, **kwargs: {})

        import scepter.gpu_accel as gpu_accel
        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _FakeGpuSession)
        perf_values = itertools.chain((0.0, 2.0, 2.0, 3.0), itertools.repeat(3.0))
        monkeypatch.setattr(nbeam.time, "perf_counter", lambda: next(perf_values))

        with h5py.File(storage, "r") as h5:
            diag = nbeam._select_auto_pure_reroute_backend(
                h5,
                ["iter_00000"],
                count_var=None,
                nco=1,
                beam_caps=np.arange(2, dtype=np.int32),
                max_demand_slots=None,
            )

        assert diag["pure_reroute_backend_selected"] == "gpu"
        assert diag["pure_reroute_probe_cpu_s"] == pytest.approx(2.0)
        assert diag["pure_reroute_probe_gpu_s"] == pytest.approx(1.0)


def test_collect_pure_reroute_probe_buffer_respects_slot_chunk_limit():
    eligible_ds = _SliceRecordingDataset(np.ones((10, 2, 3), dtype=np.bool_))
    counts_ds = _SliceRecordingDataset(np.ones((10, 2), dtype=np.int32))
    fake_h5 = {
        "iter": {
            "iter_00000": {
                "sat_eligible_mask": eligible_ds,
                "sat_beam_counts_eligible": counts_ds,
            }
        }
    }

    probe_mask, probe_slots, probe_edges = nbeam._collect_pure_reroute_probe_buffer(
        fake_h5,  # type: ignore[arg-type]
        ["iter_00000"],
        count_var="sat_beam_counts_eligible",
        target_slots=6,
        target_edges=10_000,
        max_slots=6,
        slot_chunk_limit=4,
    )

    assert probe_mask.shape == (6, 2, 3)
    assert probe_slots == 6
    assert probe_edges == 36
    assert eligible_ds.requests == [(0, 4), (4, 6)]
    assert all((stop - start) <= 4 for start, stop in eligible_ds.requests)


def test_select_auto_pure_reroute_backend_caps_probe_to_read_slot_chunk(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "auto_probe_chunk_cap.h5"
        eligible_mask = np.ones((2, 1, 1), dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)

        recorded: dict[str, int] = {}

        class _FakeGpuSession:
            def __init__(self, *args, **kwargs):
                return None

            def close(self, reset_device=False):
                return None

        def _fake_collect(*args, **kwargs):
            recorded["target_slots"] = int(kwargs["target_slots"])
            recorded["max_slots"] = int(kwargs["max_slots"])
            recorded["slot_chunk_limit"] = int(kwargs["slot_chunk_limit"])
            return eligible_mask.astype(bool, copy=False), 1, 1

        monkeypatch.setattr(nbeam, "_load_tqdm", lambda: None)
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
        monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: True)
        monkeypatch.setattr(nbeam, "READ_SLOT_CHUNK", 32)
        monkeypatch.setattr(nbeam, "_collect_pure_reroute_probe_buffer", _fake_collect)
        monkeypatch.setattr(nbeam, "_run_pure_reroute_cpu_solver", lambda *args, **kwargs: {})
        monkeypatch.setattr(nbeam, "_run_pure_reroute_gpu_solver", lambda *args, **kwargs: {})

        import scepter.gpu_accel as gpu_accel

        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _FakeGpuSession)
        perf_values = itertools.chain((0.0, 1.0, 1.0, 2.2), itertools.repeat(2.2))
        monkeypatch.setattr(nbeam.time, "perf_counter", lambda: next(perf_values))

        with h5py.File(storage, "r") as h5:
            nbeam._select_auto_pure_reroute_backend(
                h5,
                ["iter_00000"],
                count_var=None,
                nco=1,
                beam_caps=np.arange(2, dtype=np.int32),
                max_demand_slots=None,
            )

        assert recorded == {
            "target_slots": 32,
            "max_slots": 32,
            "slot_chunk_limit": 32,
        }


def test_select_auto_pure_reroute_backend_can_use_smaller_internal_probe_cap(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "auto_probe_internal_cap.h5"
        eligible_mask = np.ones((2, 1, 1), dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)

        recorded: dict[str, int] = {}

        class _FakeGpuSession:
            def __init__(self, *args, **kwargs):
                return None

            def close(self, reset_device=False):
                return None

        def _fake_collect(*args, **kwargs):
            recorded["target_slots"] = int(kwargs["target_slots"])
            recorded["max_slots"] = int(kwargs["max_slots"])
            recorded["slot_chunk_limit"] = int(kwargs["slot_chunk_limit"])
            return eligible_mask.astype(bool, copy=False), 1, 1

        monkeypatch.setattr(nbeam, "_load_tqdm", lambda: None)
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
        monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: True)
        monkeypatch.setattr(nbeam, "READ_SLOT_CHUNK", 1024)
        monkeypatch.setattr(nbeam, "_collect_pure_reroute_probe_buffer", _fake_collect)
        monkeypatch.setattr(nbeam, "_run_pure_reroute_cpu_solver", lambda *args, **kwargs: {})
        monkeypatch.setattr(nbeam, "_run_pure_reroute_gpu_solver", lambda *args, **kwargs: {})

        import scepter.gpu_accel as gpu_accel

        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _FakeGpuSession)
        perf_values = itertools.chain((0.0, 1.0, 1.0, 2.2), itertools.repeat(2.2))
        monkeypatch.setattr(nbeam.time, "perf_counter", lambda: next(perf_values))

        with h5py.File(storage, "r") as h5:
            nbeam._select_auto_pure_reroute_backend(
                h5,
                ["iter_00000"],
                count_var=None,
                nco=1,
                beam_caps=np.arange(2, dtype=np.int32),
                max_demand_slots=None,
            )

        assert recorded == {
            "target_slots": nbeam._PURE_REROUTE_AUTO_PROBE_MAX_SLOTS,
            "max_slots": nbeam._PURE_REROUTE_AUTO_PROBE_MAX_SLOTS,
            "slot_chunk_limit": nbeam._PURE_REROUTE_AUTO_PROBE_MAX_SLOTS,
        }


def test_run_pure_reroute_gpu_solver_wraps_nvrtc_header_errors():
    class _BrokenGpuSession:
        def pure_reroute_service_curve(self, *args, **kwargs):
            raise RuntimeError('cannot open source file "cuda_fp16.h"')

    with pytest.raises(RuntimeError, match="cuda_fp16.h") as excinfo:
        nbeam._run_pure_reroute_gpu_solver(
            _BrokenGpuSession(),
            np.ones((1, 1, 1), dtype=np.bool_),
            nco=1,
            beam_caps=np.arange(2, dtype=np.int32),
        )

    assert "scepter-dev-full" in str(excinfo.value)
    assert 'pure_reroute_backend="cpu"' in str(excinfo.value)


def test_run_beam_cap_sizing_auto_uses_selected_gpu_backend(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "auto_gpu.h5"
        eligible_mask = np.array([[[True, True], [True, False]]], dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)

        used_gpu_backend = {"value": False}

        class _FakeGpuSession:
            def __init__(self, *args, **kwargs):
                used_gpu_backend["value"] = True

            def pure_reroute_service_curve(self, eligible_mask, *, nco, beam_caps, return_device=False):
                assert return_device is False
                return satsim.pure_reroute_service_curve(eligible_mask, nco=nco, beam_caps=beam_caps)

            def close(self, reset_device=False):
                return None

        monkeypatch.setattr(
            nbeam,
            "_select_auto_pure_reroute_backend",
            lambda *args, **kwargs: {
                "pure_reroute_backend_requested": "auto",
                "pure_reroute_backend_selected": "gpu",
                "pure_reroute_probe_slots": 1,
                "pure_reroute_probe_edges": 3,
                "pure_reroute_probe_cpu_s": 2.0,
                "pure_reroute_probe_gpu_s": 1.0,
            },
        )

        import scepter.gpu_accel as gpu_accel
        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _FakeGpuSession)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("pure_reroute",),
                pure_reroute_backend="auto",
                nco=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert used_gpu_backend["value"] is True
        assert result["selected_caps"]["pure_reroute"] == 1
        assert result["run_diagnostics"]["pure_reroute_backend_selected"] == "gpu"


def test_run_beam_cap_sizing_explicit_gpu_requires_available_backend(monkeypatch):
    monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: False)
    with pytest.raises(RuntimeError, match="pure_reroute_backend='gpu'"):
        nbeam._resolve_pure_reroute_backend("gpu")


def test_run_beam_cap_sizing_temporal_policy_matches_expected_fixture():
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "temporal.h5"
        counts = np.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=np.int32,
        )
        belt = np.array(
            [
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
            ],
            dtype=np.int16,
        )
        az = np.array(
            [
                [[0.0], [180.0]],
                [[6.0], [180.0]],
                [[0.0], [180.0]],
            ],
            dtype=np.float32,
        )
        el = np.array(
            [
                [[30.0], [30.0]],
                [[30.0], [30.0]],
                [[30.0], [30.0]],
            ],
            dtype=np.float32,
        )
        times = (np.arange(3, dtype=np.float64)[:, np.newaxis]) / 86400.0
        _write_iter_group(storage, counts=counts, belt=belt, az=az, el=el, times=times)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("belt_sky_temporal",),
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert result["selected_caps"]["belt_sky_temporal"] == 1
        curve = result["policy_curves"]["belt_sky_temporal"]
        np.testing.assert_allclose(curve["delta"][:2], np.array([1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(curve["eps"][:2], np.array([1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)


def test_run_beam_cap_sizing_interim_json_creates_run_dir_and_glossary_payload():
    with TemporaryDirectory(dir=".") as tmp_dir:
        tmp_path = Path(tmp_dir)
        storage = tmp_path / "interim_json.h5"
        counts = np.array(
            [
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        )
        _write_iter_group(storage, counts=counts)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute",),
                save_outputs=False,
                save_interim_outputs=True,
                interim_every_slots=5,
                interim_save_html=False,
                interim_save_json=True,
                read_slot_chunk=4,
                output_dir=tmp_path / "out",
                output_prefix="interim",
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        run_dir = Path(result["output_run_dir"])
        payload = json.loads((run_dir / "results_summary.interim.json").read_text(encoding="utf-8"))

        assert run_dir.exists()
        assert payload["snapshot"]["is_final"] is False
        assert payload["snapshot"]["is_partial"] is True
        assert payload["snapshot"]["raw_slots_seen"] == 9
        assert payload["snapshot"]["raw_slots_total"] == 9
        assert payload["snapshot"]["demand_slots"] == 2
        assert payload["run_diagnostics"]["interim_snapshots_written"] == 2
        assert payload["metric_glossary"]["delta"]["definition"] == (
            "delta(B) = total unserved demand divided by total processed demand"
        )
        assert payload["metric_glossary"]["epsilon"]["definition"] == (
            "epsilon(B) / eps(B) = fraction of processed demand slots that still fail full service"
        )
        assert payload["metric_glossary"]["tail_risk"]["definition"] == (
            "tail risk = P(B_req > B), the fraction of processed slots whose required beam cap exceeds B"
        )


def test_run_beam_cap_sizing_interim_cadence_uses_raw_slots_and_is_chunk_aligned(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "interim_cadence.h5"
        counts = np.array(
            [
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        )
        _write_iter_group(storage, counts=counts)

        captured_snapshots: list[tuple[int, int, bool]] = []

        def _fake_write_snapshot_artifacts(**kwargs):
            snapshot = kwargs["summary_payload"]["snapshot"]
            captured_snapshots.append(
                (int(snapshot["raw_slots_seen"]), int(snapshot["demand_slots"]), bool(snapshot["is_final"]))
            )

        monkeypatch.setattr(nbeam, "_write_snapshot_artifacts", _fake_write_snapshot_artifacts)

        run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute",),
                save_outputs=False,
                save_interim_outputs=True,
                interim_every_slots=5,
                interim_save_html=False,
                interim_save_json=True,
                read_slot_chunk=4,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert captured_snapshots == [(8, 2, False), (9, 2, False)]


def test_run_beam_cap_sizing_interim_html_reuses_stable_paths(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "interim_html.h5"
        counts = np.ones((12, 2), dtype=np.int32)
        _write_iter_group(storage, counts=counts)

        html_paths: list[str] = []

        def _fake_write_html(self, path, include_plotlyjs="cdn"):
            html_paths.append(str(path))

        monkeypatch.setattr(nbeam.go.Figure, "write_html", _fake_write_html)

        run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute",),
                save_outputs=False,
                save_interim_outputs=True,
                interim_every_slots=5,
                interim_save_html=True,
                interim_save_json=False,
                read_slot_chunk=4,
                output_dir=Path(tmp_dir) / "out",
                output_prefix="htmlsnap",
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        names = [Path(path).name for path in html_paths]
        assert names.count("tail_risk.interim.html") == 2
        assert names.count("sla_delta.interim.html") == 2
        assert names.count("sla_eps.interim.html") == 2
        assert sorted(set(names)) == [
            "sla_delta.interim.html",
            "sla_eps.interim.html",
            "tail_risk.interim.html",
        ]


def test_run_beam_cap_sizing_final_outputs_keep_final_filenames_with_interim_enabled():
    with TemporaryDirectory(dir=".") as tmp_dir:
        tmp_path = Path(tmp_dir)
        storage = tmp_path / "final_and_interim.h5"
        counts = np.ones((8, 2), dtype=np.int32)
        _write_iter_group(storage, counts=counts)

        result = run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute",),
                save_outputs=True,
                save_plots_html=False,
                save_results_json=True,
                save_results_npz=False,
                save_interim_outputs=True,
                interim_every_slots=5,
                interim_save_html=False,
                interim_save_json=True,
                read_slot_chunk=4,
                output_dir=tmp_path / "out",
                output_prefix="both",
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        run_dir = Path(result["output_run_dir"])
        final_payload = json.loads((run_dir / "results_summary.json").read_text(encoding="utf-8"))
        interim_payload = json.loads((run_dir / "results_summary.interim.json").read_text(encoding="utf-8"))

        assert (run_dir / "results_summary.json").exists()
        assert (run_dir / "results_summary.interim.json").exists()
        assert final_payload["snapshot"]["is_final"] is True
        assert final_payload["snapshot"]["is_partial"] is False
        assert interim_payload["snapshot"]["is_final"] is False
        assert interim_payload["snapshot"]["raw_slots_seen"] == 8


def test_run_beam_cap_sizing_explicit_gpu_preflights_and_raises_actionable_error(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "explicit_gpu_nvrtc_error.h5"
        eligible_mask = np.ones((2, 1, 1), dtype=bool)
        _write_iter_group(storage, eligible_mask=eligible_mask)

        messages: list[str] = []

        class _BrokenGpuSession:
            def __init__(self, *args, **kwargs):
                return None

            def pure_reroute_service_curve(self, *args, **kwargs):
                raise RuntimeError('cannot open source file "cuda_fp16.h"')

            def close(self, reset_device=False):
                return None

        monkeypatch.setattr(nbeam, "_load_tqdm", lambda: None)
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: messages.append(" ".join(str(arg) for arg in args)))
        monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: True)

        import scepter.gpu_accel as gpu_accel

        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _BrokenGpuSession)

        with pytest.raises(RuntimeError, match="cuda_fp16.h") as excinfo:
            run_beam_cap_sizing(
                storage,
                config=BeamCapSizingConfig(
                    enabled_policy_keys=("pure_reroute",),
                    pure_reroute_backend="gpu",
                    nco=1,
                    read_slot_chunk=4,
                    save_outputs=False,
                    show_plots=False,
                    enable_skycell_demand_vis=False,
                ),
            )

        assert "scepter-dev-full" in str(excinfo.value)
        assert any("[startup] Preflighting pure-reroute GPU solver." in msg for msg in messages)


def test_run_beam_cap_sizing_reports_stage_order_with_tqdm(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "tqdm_stage_reporting.h5"
        eligible_mask = np.ones((2, 2, 2), dtype=bool)
        counts = np.array([[1, 1], [1, 0]], dtype=np.int32)
        _write_iter_group(storage, eligible_mask=eligible_mask, counts=counts)

        events: list[str] = []
        bar_creations: list[tuple[str, int | None, str | None]] = []

        class _FakeTqdm:
            def __init__(self, *args, **kwargs):
                self.desc = kwargs.get("desc")
                self.total = kwargs.get("total")
                self.unit = kwargs.get("unit")
                bar_creations.append((str(self.desc), None if self.total is None else int(self.total), self.unit))

            @staticmethod
            def write(message):
                events.append(str(message))

            def update(self, delta):
                return None

            def set_postfix(self, values, refresh=False):
                return None

            def refresh(self):
                return None

            def close(self):
                return None

        class _FakeGpuSession:
            def __init__(self, *args, **kwargs):
                return None

            def close(self, reset_device=False):
                return None

        def _fake_collect(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback is not None:
                progress_callback(2, 2, 8)
            return np.ones((2, 2, 2), dtype=np.bool_), 2, 8

        monkeypatch.setattr(nbeam, "_ACTIVE_PROGRESS_BAR_COUNT", 0)
        monkeypatch.setattr(nbeam, "_load_tqdm", lambda: _FakeTqdm)
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: events.append(" ".join(str(arg) for arg in args)))
        monkeypatch.setattr(nbeam, "_pure_reroute_gpu_backend_available", lambda: True)
        monkeypatch.setattr(nbeam, "_collect_pure_reroute_probe_buffer", _fake_collect)
        monkeypatch.setattr(
            nbeam,
            "_run_pure_reroute_cpu_solver",
            lambda eligible_chunk, *, nco, beam_caps: satsim.pure_reroute_service_curve(
                eligible_chunk,
                nco=nco,
                beam_caps=beam_caps,
            ),
        )
        monkeypatch.setattr(
            nbeam,
            "_run_pure_reroute_gpu_solver",
            lambda session, eligible_chunk, *, nco, beam_caps: satsim.pure_reroute_service_curve(
                eligible_chunk,
                nco=nco,
                beam_caps=beam_caps,
            ),
        )

        import scepter.gpu_accel as gpu_accel

        monkeypatch.setattr(gpu_accel, "GpuScepterSession", _FakeGpuSession)
        perf_values = itertools.chain((0.0, 1.0, 1.0, 2.2), itertools.repeat(2.2))
        monkeypatch.setattr(nbeam.time, "perf_counter", lambda: next(perf_values))

        run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute", "pure_reroute"),
                pure_reroute_backend="auto",
                read_slot_chunk=4,
                progress_every_demand_slots=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        startup_idx = next(i for i, msg in enumerate(events) if "[startup] Preparing beam-cap sizing run." in msg)
        dataset_idx = next(i for i, msg in enumerate(events) if "[dataset-inspection] Inspecting HDF5 dataset layout." in msg)
        probe_idx = next(i for i, msg in enumerate(events) if "[pure-reroute-probe] Benchmarking exact CPU vs GPU solver" in msg)

        assert startup_idx < dataset_idx < probe_idx
        assert ("[pure-reroute-probe] active slots", 4, "slot") in bar_creations
        assert ("[main-stream] raw slots", 2, "slot") in bar_creations


def test_run_beam_cap_sizing_falls_back_to_text_reporting_when_tqdm_unavailable(monkeypatch):
    with TemporaryDirectory(dir=".") as tmp_dir:
        storage = Path(tmp_dir) / "text_stage_reporting.h5"
        counts = np.array([[1, 0], [0, 1]], dtype=np.int32)
        _write_iter_group(storage, counts=counts)

        messages: list[str] = []
        monkeypatch.setattr(nbeam, "_ACTIVE_PROGRESS_BAR_COUNT", 0)
        monkeypatch.setattr(nbeam, "_load_tqdm", lambda: None)
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: messages.append(" ".join(str(arg) for arg in args)))

        run_beam_cap_sizing(
            storage,
            config=BeamCapSizingConfig(
                enabled_policy_keys=("full_reroute",),
                read_slot_chunk=1,
                progress_every_demand_slots=1,
                save_outputs=False,
                show_plots=False,
                enable_skycell_demand_vis=False,
            ),
        )

        assert any("[startup] Preparing beam-cap sizing run." in msg for msg in messages)
        assert any("[main-stream] raw slots:" in msg for msg in messages)
        assert any("[main-stream] Streaming pass completed." in msg for msg in messages)


def test_pure_reroute_dataset_diagnostics_warns_for_large_chunk_estimates():
    ds = _LayoutOnlyDataset(
        shape=(1024, 4096, 4096),
        dtype=np.bool_,
        chunks=(2048, 1, 1),
        compression="gzip",
        shuffle=True,
    )

    diag = nbeam._pure_reroute_dataset_diagnostics(
        ds,  # type: ignore[arg-type]
        read_slot_chunk=128,
        max_demand_slots=None,
    )

    assert any("main chunk exceeds 1 GiB" in warning for warning in diag["warnings"])
    assert any("auto-probe chunk exceeds 1 GiB" in warning for warning in diag["warnings"])
    assert any("leading HDF5 chunk spans" in warning for warning in diag["warnings"])


def test_beam_cap_metric_glossary_is_documented_consistently():
    glossary_terms = [
        ("delta(B)", "total unserved demand", "total processed demand"),
        ("epsilon(B)", "processed demand slots", "full service"),
        ("P(B_req > B)", "processed slots", "beam cap exceeds"),
    ]

    _doc_files = [
        REPO_ROOT / "NBeamParserAdvanced-NEO.py",
        REPO_ROOT / "NBeamParserNEO.ipynb",
        REPO_ROOT / "README.md",
    ]
    for _df in _doc_files:
        if not _df.is_file():
            pytest.skip(f"Documentation file {_df.name} not present in this checkout.")
    doc_texts = [
        BeamCapSizingConfig.__doc__ or "",
        run_beam_cap_sizing.__doc__ or "",
        _doc_files[0].read_text(encoding="utf-8"),
        _doc_files[1].read_text(encoding="utf-8"),
        _doc_files[2].read_text(encoding="utf-8"),
    ]
    normalized_texts = [
        " ".join(text.replace("#", " ").replace("\\n", " ").replace('"', " ").split())
        for text in doc_texts
    ]

    for terms in glossary_terms:
        assert all(all(term in text for term in terms) for text in normalized_texts)

    low_memory_terms = [
        ("READ_SLOT_CHUNK", "pure_reroute", "tqdm.auto"),
        ("READ_SLOT_CHUNK", "hard upper bound", "plain text"),
    ]

    for terms in low_memory_terms:
        assert all(all(term in text for term in terms) for text in normalized_texts)
