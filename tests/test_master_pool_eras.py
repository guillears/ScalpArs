"""Master-pool era registry + auto-discovery (Sep-11): B3/B4/B5 must be first-class eras and any
later archive reports/BASELINE<n>_*.csv (n ≥ 6) must be picked up automatically — exactly once.
Falsifiable: dropping an era from FIXED_ERAS, or letting two BASELINE6 files through, fails."""
import importlib.util, os, pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_master_pool", _ROOT / "scripts" / "build_master_pool.py")
bmp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bmp)


def test_fixed_eras_cover_base_through_b5_in_order():
    assert [e[0] for e in bmp.FIXED_ERAS] == ['BASE', 'B1', 'B2', 'B3', 'B4', 'B5']
    for _, path, _, _ in bmp.FIXED_ERAS:
        assert (_ROOT / path).exists(), path


def test_discovery_adds_b6_plus_from_archive_naming(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "BASELINE6_batch0911-0930_orders_prereset.csv").write_text("x\n")
    (tmp_path / "reports" / "BASELINE7_batch1001_orders.csv").write_text("x\n")
    (tmp_path / "reports" / "BASELINE6_batch0911-0930_split_report.txt").write_text("x\n")  # ignored
    (tmp_path / "reports" / "BASELINE5_batch0826-0903_orders.csv").write_text("x\n")         # n<6 → fixed list only
    monkeypatch.chdir(tmp_path)
    eras = bmp.discover_eras()
    assert [e[0] for e in eras][-2:] == ['B6', 'B7']
    assert eras[-2][1].endswith("BASELINE6_batch0911-0930_orders_prereset.csv")
    assert sum(1 for e in eras if e[0] == "B5") == 1


def test_two_archives_for_one_era_is_fatal(tmp_path, monkeypatch):
    import pytest
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "BASELINE6_a_orders.csv").write_text("x\n")
    (tmp_path / "reports" / "BASELINE6_b_orders.csv").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        bmp.discover_eras()


def test_only_b1_is_lenient_on_null_status():
    lenient = {e[0] for e in bmp.FIXED_ERAS if e[3]}
    assert lenient == {'B1'}
