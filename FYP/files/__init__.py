"""
UWB Indoor Positioning System - Modular Dashboard
===================================================
EE4002D Final Year Project - Wayne Wong


Architecture:
  UWB Anchors <-ranging-> Tag -> WiFi/MQTT -> Server -> SSE -> Web Dashboard

Modules:
  config.py       - System configuration (room, anchors, MQTT, smoothing)
  models.py       - Data structures (Position, GroundTruth, AlgorithmResult)
  algorithms.py   - Positioning algorithms (WCL, LS, BCCP, Fused)
  engine.py       - Positioning engine (distance processing, smoothing, error tracking)
  experiment.py   - Experiment manager (test point queue, recording, CSV/JSON export)
  mqtt_handler.py - MQTT client for receiving UWB distance data
  server.py       - FastAPI application + API routes + SSE streaming
  run.py          - Entry point
"""
