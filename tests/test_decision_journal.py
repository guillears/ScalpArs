"""📓 Decision journal — must be file-only, never raise, honour the off switches, and keep its files tidy."""
import gzip
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import config
from services import decision_journal as dj


def _setup(tmp_path, monkeypatch, enabled=True, keep=45):
    monkeypatch.setattr(dj, '_dir', lambda: str(tmp_path / 'journal'))
    monkeypatch.delenv('SCALPARS_JOURNAL_OFF', raising=False)
    monkeypatch.setattr(config.trading_config, 'thresholds',
                        SimpleNamespace(decision_journal_enabled=enabled, decision_journal_retention_days=keep), raising=False)
    dj._buffer[:] = []
    dj._state['day'] = None


def test_writes_one_json_line_per_event(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    dj.note('SCAN', btc_rsi=71.234567, veto_long='BTC_RSI_ADX_CROSS', nothing=None)
    dj.note('BLOCK', gate='BTC_ADX_GATE_HIGH', dir='LONG', pair='SOLUSDT', ctx=dj.snapshot({'rsi': 61.23456, 'adx': float('nan'), 'junk': 1}))
    dj.flush()
    files = os.listdir(tmp_path / 'journal')
    assert len(files) == 1 and files[0].startswith('decisions-') and files[0].endswith('.jsonl')
    rows = [json.loads(l) for l in open(tmp_path / 'journal' / files[0])]
    assert [r['e'] for r in rows] == ['SCAN', 'BLOCK']
    assert rows[0]['btc_rsi'] == 71.234567 and 'nothing' not in rows[0]
    assert rows[1]['ctx'] == {'rsi': 61.23456, 'adx': None}       # NaN → null, unknown keys dropped
    dj.note('OPEN', pair='1000PEPEUSDT', price=0.0038596123456); dj.flush()
    assert json.loads(open(tmp_path / 'journal' / files[0]).read().splitlines()[-1])['price'] == 0.003859612346   # 10 significant digits, not 4 dp


def test_disabled_and_env_switch_write_nothing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, enabled=False)
    dj.note('OPEN', pair='X'); dj.flush()
    assert not (tmp_path / 'journal').exists()
    _setup(tmp_path, monkeypatch, enabled=True)
    monkeypatch.setenv('SCALPARS_JOURNAL_OFF', '1')
    dj.note('OPEN', pair='X'); dj.flush()
    assert not (tmp_path / 'journal').exists()


def test_never_raises_when_the_folder_is_unwritable(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    blocker = tmp_path / 'journal'
    blocker.write_text('i am a file, not a folder')           # makedirs/open will fail
    dj.note('OPEN', pair='X', weird=object())
    dj.flush()                                                 # must swallow
    assert dj._buffer == []                                    # and must not grow without bound


def test_rollover_gzips_finished_days_and_deletes_old_ones(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, keep=10)
    d = tmp_path / 'journal'; d.mkdir()
    old = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
    yday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    (d / f'decisions-{old}.jsonl.gz').write_bytes(b'x')
    (d / f'decisions-{yday}.jsonl').write_text('{"e":"SCAN"}\n')
    dj.note('SCAN'); dj.flush()
    names = sorted(os.listdir(d))
    assert f'decisions-{old}.jsonl.gz' not in names
    assert f'decisions-{yday}.jsonl.gz' in names and f'decisions-{yday}.jsonl' not in names
    assert gzip.open(d / f'decisions-{yday}.jsonl.gz', 'rt').read() == '{"e":"SCAN"}\n'


def test_buffer_is_bounded(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    for i in range(dj._MAX_BUFFER + 5):
        dj.note('BLOCK', gate='G', pair=f'P{i}')
    assert len(dj._buffer) < dj._MAX_BUFFER                      # auto-flushed at the cap
