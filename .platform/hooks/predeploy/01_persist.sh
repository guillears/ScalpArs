#!/bin/bash
# Ensure persistent data directory exists with proper permissions
# so the webapp user can read/write the SQLite DB and config.
#
# NOTE: On Amazon Linux 2023, EB uses $EB_APP_STAGING_DIR to point to the
# staging directory, which may differ from /var/app/staging. We use that
# env var when available and fall back to the hardcoded path otherwise.
#
# Also note: the SQLite DB path is now set to an ABSOLUTE path in config.py
# (/opt/scalpars-data/scalpars.db) so the symlink below is defensive only —
# the Python app connects directly to the persistent file regardless of
# what's in /var/app/current/. This means data survives deploys even if
# the staging symlink mechanism breaks again.

set -e

STAGING_DIR="${EB_APP_STAGING_DIR:-/var/app/staging}"
PERSISTENT_DIR="/opt/scalpars-data"

mkdir -p "$PERSISTENT_DIR"
chmod 777 "$PERSISTENT_DIR"

# Create the DB file if it doesn't exist (SQLite can't create through a dangling symlink)
if [ ! -f "$PERSISTENT_DIR/scalpars.db" ]; then
  touch "$PERSISTENT_DIR/scalpars.db"
fi
chmod 666 "$PERSISTENT_DIR/scalpars.db"

# Always deploy the latest trading_config.json from git to persistent storage
# (matches existing behavior: UI settings are overwritten on deploy unless
#  the user also commits them to git — documented in CLAUDE.md).
if [ -f "$STAGING_DIR/trading_config.json" ]; then
  cp "$STAGING_DIR/trading_config.json" "$PERSISTENT_DIR/trading_config.json" 2>/dev/null || true
  chmod 666 "$PERSISTENT_DIR/trading_config.json" 2>/dev/null || true
fi

# Symlink staging → persistent (defensive; config.py now uses absolute path for DB)
ln -sf "$PERSISTENT_DIR/scalpars.db" "$STAGING_DIR/scalpars.db"
ln -sf "$PERSISTENT_DIR/trading_config.json" "$STAGING_DIR/trading_config.json"

echo "[PREDEPLOY_HOOK] persistent dir=$PERSISTENT_DIR, staging=$STAGING_DIR"
ls -la "$PERSISTENT_DIR/" || true
