"""Live Terminal — UI-only observability layer over the bot's stdlib logging.

This package is purely additive. It attaches a logging.Handler to the root
logger to capture every log record into a ring buffer, then exposes the
buffer via Server-Sent Events for a dedicated terminal-style UI view.

It MUST NOT modify or import any trading-logic code in services/.
The only services module read here is services.trading_engine — and only
its read-only module-level globals (regime, breadth, btc_adx, etc.) for
the synthetic [HEARTBEAT] event. No function calls into trading logic.
"""
