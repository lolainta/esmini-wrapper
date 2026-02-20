#!/bin/bash
pushd /app
uv run esmini_wrapper/server.py
popd
