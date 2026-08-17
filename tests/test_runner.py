import pytest

from inference_bench.runner import load_backend_specs


def test_load_backend_specs_defaults_pid_to_none_when_absent(tmp_path):
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        """
backends:
  - name: no-pid
    kind: mock
    tokens_per_s: 80
"""
    )
    specs = load_backend_specs(config_path)
    assert len(specs) == 1
    _, spec, _ = specs[0]
    assert spec.pid is None


def test_load_backend_specs_parses_pid_when_present(tmp_path):
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        """
backends:
  - name: with-pid
    kind: mock
    tokens_per_s: 80
    pid: 4242
"""
    )
    specs = load_backend_specs(config_path)
    _, spec, _ = specs[0]
    assert spec.pid == 4242


def test_load_backend_specs_defaults_gpu_vendor_to_nvidia_when_absent(tmp_path):
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        """
backends:
  - name: no-vendor
    kind: mock
    tokens_per_s: 80
    pid: 4242
"""
    )
    specs = load_backend_specs(config_path)
    _, spec, _ = specs[0]
    assert spec.gpu_vendor == "nvidia"


def test_load_backend_specs_parses_amd_gpu_vendor(tmp_path):
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        """
backends:
  - name: amd-backend
    kind: mock
    tokens_per_s: 80
    pid: 4242
    gpu_vendor: amd
"""
    )
    specs = load_backend_specs(config_path)
    _, spec, _ = specs[0]
    assert spec.gpu_vendor == "amd"


def test_load_backend_specs_rejects_unknown_gpu_vendor(tmp_path):
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        """
backends:
  - name: bad-vendor
    kind: mock
    tokens_per_s: 80
    pid: 4242
    gpu_vendor: intel
"""
    )
    with pytest.raises(ValueError, match="gpu_vendor"):
        load_backend_specs(config_path)
