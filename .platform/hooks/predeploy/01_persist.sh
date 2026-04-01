#!/bin/bash
# Ensure persistent data directory exists and link files so SQLite DB
# and trading_config.json survive across deployments.
mkdir -p /opt/scalpars-data

# If this is a fresh environment (no DB yet), the app will create it.
# If trading_config.json doesn't exist in persistent storage yet, copy the repo version.
if [ ! -f /opt/scalpars-data/trading_config.json ]; then
  cp /var/app/staging/trading_config.json /opt/scalpars-data/trading_config.json 2>/dev/null || true
fi

# Create symlinks so the app reads/writes to persistent storage
ln -sf /opt/scalpars-data/scalpars.db /var/app/staging/scalpars.db
ln -sf /opt/scalpars-data/trading_config.json /var/app/staging/trading_config.json
