#!/bin/bash
# Ensure persistent data directory exists with proper permissions
# so the webapp user can read/write the SQLite DB and config.
mkdir -p /opt/scalpars-data
chmod 777 /opt/scalpars-data

# Create the DB file if it doesn't exist (SQLite can't create through a dangling symlink)
if [ ! -f /opt/scalpars-data/scalpars.db ]; then
  touch /opt/scalpars-data/scalpars.db
fi
chmod 666 /opt/scalpars-data/scalpars.db

# Copy trading_config.json on first deploy, then always symlink to persistent copy
if [ ! -f /opt/scalpars-data/trading_config.json ]; then
  cp /var/app/staging/trading_config.json /opt/scalpars-data/trading_config.json 2>/dev/null || true
fi
chmod 666 /opt/scalpars-data/trading_config.json 2>/dev/null || true

# Symlink so the app reads/writes to persistent storage
ln -sf /opt/scalpars-data/scalpars.db /var/app/staging/scalpars.db
ln -sf /opt/scalpars-data/trading_config.json /var/app/staging/trading_config.json
