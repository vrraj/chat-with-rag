.PHONY: start stop rebuild start-hybrid stop-hybrid bg-start-chat start-debug start-docker start-qdrant stop-qdrant qdrant-logs qdrant-restart qdrant-status qdrant-collections qdrant-info qdrant-info-json qdrant-indexes smoke_api

# Qdrant endpoint configuration. Prefer environment overrides; else fall back to backend Settings; else sensible defaults
ifndef QDRANT_HOST	
QDRANT_HOST := $(shell python3 -c "from backend.core.config import Settings; print(Settings().qdrant_host)" 2>/dev/null || echo localhost)
endif
ifndef QDRANT_PORT
QDRANT_PORT := $(shell python3 -c "from backend.core.config import Settings; print(Settings().qdrant_port)" 2>/dev/null || echo 6333)
endif
UNAME := $(shell uname)

# Start the full application stack using Docker Compose
# Qdrant database runs in a Docker container
# Web application runs in a container with local code bind-mounted
# Use run.py  in docker-compose.yml for auto app code reload, and start.py to run in production
# Prerequisites: Docker and Docker Compose must be installed
# Usage: make start
start:
	@echo "Starting Qdrant and Chat application..."
	@echo "Detected OS: $(UNAME)"

	# Start Docker Desktop on macOS if not running
	@if [ "$(UNAME)" = "Darwin" ]; then \
		echo "Opening Docker Desktop GUI on Mac."; \
			open -a Docker || open -a "Docker Desktop"; \
		else \
		echo "No GUI launch command for non-Darwin OS."; \
	fi

	# Wait for Docker daemon
	@echo "Waiting for Docker daemon to become available..."
	@TIMEOUT=30; \
	while ! docker info >/dev/null 2>&1; do \
		if [ $$TIMEOUT -le 0 ]; then \
			echo "Error: Docker daemon did not start within 30 seconds. Please check Docker Desktop."; \
			exit 1; \
		fi; \
		printf '.'; \
		sleep 1; \
		TIMEOUT=$$(($$TIMEOUT - 1)); \
	done; \
	echo ""; \
	echo "Docker daemon is running. Proceeding with compose."

	# Start the services
	@if ! docker compose up -d; then \
		echo "Error: Failed to start services with docker compose"; \
		exit 1; \
	fi

	@echo "Qdrant and Chat application started successfully."
	@echo "Access the application at: http://localhost:8000"
	@echo "Qdrant and Chat application started successfully."

# Stop the application - all Docker containers
stop:
	echo "Stopping Qdrant and Chat application..."
	docker compose down
	echo "Qdrant and Chat application stopped successfully."

# Rebuild and start the application with latest code changes
rebuild:
	docker compose up -d --build

# Alternative development mode (hybrid)
# - Qdrant database runs in a Docker container
# - Web application runs in a local Python virtual environment (outside Docker)
# - Uses run.py with auto-reload enabled for development
# Use this if you prefer running the web app natively or have specific development needs
# Prerequisites: Python 3.8+, virtualenv, and Docker must be installed
# Usage: make start-hybrid
start-hybrid:
	echo "Starting Qdrant and Chat application in hybrid mode..."
	@$(MAKE) start-docker
	@$(MAKE) start-qdrant
	@. venv/bin/activate && $(MAKE) bg-start-chat
	echo "Qdrant and Chat application started successfully."

# Quick Stop the Chat application - includes qdrant, activating venv and running uvicorn
stop-hybrid:
	echo "Stopping Qdrant and Chat application..."
	@$(MAKE) stop-qdrant
	@echo "Qdrant Database service stopped"
	@$(MAKE) stop-uvicorn
	@echo "Application stopped" 
	@echo "To shut down docker, stop it from the Docker Desktop application."

# Run FastAPI server in BACKGROUND with auto-reload enabled for development and loggong config managed in logging.py 
# Use this if you prefer running the web app natively.
bg-start-chat:
	@echo "Starting Chat backend..."
	nohup python run.py > /dev/null 2>&1 &
	@echo "Chat backend started in background."


# Run FastAPI server in foreground with live reload
# For Debugging uvicorn logs (only when developing in venv)
start-debug:
	@echo "Activating virtual environment and starting FastAPI server..."
	@. venv/bin/activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug


# Docker commands to start the docker desktop 
start-docker:
	open -a Docker
	@echo " Starting Docker Desktop..."
	@while ! docker info > /dev/null 2>&1; do \
		echo "⏳ Waiting for Docker to start..."; \
		sleep 2; \
	done
	@echo " Docker is ready!"

# Qdrant related targets
# Start Qdrant in detached (background) mode
start-qdrant:
	@echo "Starting Qdrant vector database..."
	docker compose up qdrant -d
	@echo "Qdrant is running"

# Stop and remove Qdrant container and resources
stop-qdrant:
	docker compose down qdrant
	@echo "Qdrant is shut down"

# Stream Qdrant logs live
qdrant-logs:
	docker compose logs -f qdrant

# Restart Qdrant (stop + start)
qdrant-restart: stop-qdrant  start-qdrant

# Check if Qdrant is running
qdrant-status:
	docker ps | grep qdrant || echo "Qdrant is not running"

# List all Qdrant collections (pretty-printed if jq or Python is available)
qdrant-collections:
	@base="http://$(QDRANT_HOST):$(QDRANT_PORT)"; \
	echo "Listing Qdrant collections from $$base/collections"; \
	resp=$$(curl -s "$$base/collections"); \
	if command -v jq >/dev/null 2>&1; then \
		echo "$$resp" | jq .; \
	else \
		echo "$$resp" | python3 -m json.tool 2>/dev/null || echo "$$resp"; \
	fi

# Show concise info for a single collection (dims, vectors, status)
# Usage: make qdrant-info COLLECTION=my_collection
qdrant-info:
	@if [ -z "$(COLLECTION)" ]; then \
		echo "Usage: make qdrant-info COLLECTION=<name>"; \
		exit 1; \
	fi
	@base="http://$(QDRANT_HOST):$(QDRANT_PORT)"; \
	info_url="$$base/collections/$(COLLECTION)"; \
	count_url="$$base/collections/$(COLLECTION)/points/count"; \
	echo "GET $$info_url"; \
	info_json=$$(curl -s "$$info_url"); \
	count_json=$$(curl -s -X POST -H 'Content-Type: application/json' -d '{"exact":true}' "$$count_url"); \
	if command -v jq >/dev/null 2>&1; then \
		status=$$(echo "$$info_json" | jq -r '.result.status // "unknown"'); \
		dims=$$(echo "$$info_json" | jq -r 'if .result.config.params.vectors.size then (.result.config.params.vectors.size|tostring) else (.result.config.params.vectors.map|to_entries|map("\(.key)=\(.value.size)")|join(", ")) end // "unknown"'); \
		vectors=$$(echo "$$count_json" | jq -r '.result.count // 0'); \
		echo "Collection: $(COLLECTION)"; \
		echo "Status:     $$status"; \
		echo "Dims:       $$dims"; \
		echo "Vectors:    $$vectors"; \
		else \
			python3 -c 'import json,sys;\ninfo=json.loads(sys.argv[1]) if len(sys.argv)>1 else {};\ncnt=json.loads(sys.argv[2]) if len(sys.argv)>2 else {};\nstatus=(info.get("result") or {}).get("status") or "unknown";\nvc=(((info.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors") or {};\nif isinstance(vc,dict):\n    if "size" in vc: dims=str(vc.get("size"));\n    elif isinstance(vc.get("map"),dict): dims=", ".join([f"{k}={v.get('size')}" for k,v in vc["map"].items() if isinstance(v,dict)]) or "unknown";\n    else: dims="unknown";\nelse: dims="unknown";\nvectors=(cnt.get("result") or {}).get("count") or 0;\nprint(f"Collection: $(COLLECTION)");\nprint(f"Status:     {status}");\nprint(f"Dims:       {dims}");\nprint(f"Vectors:    {vectors}")' "$$info_json" "$$count_json" \
			|| { echo; echo "[jq/python unavailable] Raw JSON follows:"; echo "$$info_json"; echo "$$count_json"; }; \
		fi

# Raw JSON info for a collection (helper)
qdrant-info-json:
	@if [ -z "$(COLLECTION)" ]; then \
		echo "Usage: make qdrant-info-json COLLECTION=<name>"; \
		exit 1; \
	fi
	@url="http://$(QDRANT_HOST):$(QDRANT_PORT)/collections/$(COLLECTION)"; \
	resp=$$(curl -s "$$url"); \
	if command -v jq >/dev/null 2>&1; then echo "$$resp" | jq .; else echo "$$resp" | python3 -m json.tool 2>/dev/null || echo "$$resp"; fi

# Show payload indexes for a collection (field name, type, points)
# Usage: make qdrant-indexes COLLECTION=my_collection
qdrant-indexes:
	@if [ -z "$(COLLECTION)" ]; then \
		echo "Usage: make qdrant-indexes COLLECTION=<name>"; \
		exit 1; \
	fi
	@base="http://$(QDRANT_HOST):$(QDRANT_PORT)"; \
	url="$$base/collections/$(COLLECTION)"; \
	echo "GET $$url"; \
	json=$$(curl -s "$$url"); \
	if command -v jq >/dev/null 2>&1; then \
		echo "Indexes for $(COLLECTION):"; \
		echo "$$json" | jq -r '(.result.payload_schema // {}) | to_entries | if length==0 then "(none)" else .[] | "- \(.key): type=\(.value.data_type), points=\(.value.points // 0)" end'; \
	else \
		python3 -c 'import json,sys; raw=sys.argv[1] if len(sys.argv)>1 else "{}"; d=json.loads(raw) if raw else {}; s=((d.get("result") or {}).get("payload_schema")) or {};\
print("(none)") if not s else [print("- {}: type={}, points={}".format(k, (v or {}).get("data_type","unknown"), (v or {}).get("points",0))) for k,v in s.items()]' "$$json"; \
	fi

# --- Qdrant storage safety & backups ---

# Backup when using bind mount ./qdrant_storage
qdrant-backup:
	@mkdir -p backups
	@tar -czf backups/qdrant-$(shell date +%F).tar.gz -C ./qdrant_storage . || echo "Nothing to back up yet (./qdrant_storage is empty?)"
	@echo " Backup written to backups/"

# Backup an old/existing named volume (e.g., qdrant_data)
qdrant-backup-volume:
	@if [ -z "$(VOLUME)" ]; then echo "Usage: make qdrant-backup-volume VOLUME=qdrant_data"; exit 1; fi
	@mkdir -p backups
	@docker run --rm -v $(VOLUME):/from -v $(PWD)/backups:/to alpine \
	  sh -lc 'cd /from && tar -czf /to/qdrant-$(VOLUME)-$$(date +%F).tar.gz .' \
	  && echo " Backup written to backups/"

# Migrate data from a named volume into the new bind mount folder
qdrant-migrate-from-volume:
	@if [ -z "$(VOLUME)" ]; then echo "Usage: make qdrant-migrate-from-volume VOLUME=qdrant_data"; exit 1; fi
	@mkdir -p qdrant_storage
	@docker-compose stop qdrant >/dev/null 2>&1 || true
	@docker run --rm -v $(VOLUME):/from -v $(PWD)/qdrant_storage:/to alpine \
	  sh -lc 'cp -a /from/. /to/' \
	  && echo " Migrated data from volume '$(VOLUME)' to ./qdrant_storage"
	@echo "Now run: docker-compose up -d && make qdrant-collections"


# Gracefully Kill processes related to uvicorn. If the application is stuck

stop-uvicorn:
	@echo "Killing (-15 - SIGTERM for clean exit) any running uvicorn processes..."
	@pkill -15 -f "uvicorn" || echo "No uvicorn process found"
 	lsof -ti :8000 | xargs kill -15


# Kill processes related to uvicorn. If the application is stuck

kill-uvicorn:
	@echo "Killing (-9 - FORCED) any running uvicorn processes..."
	@pkill -9 -f "uvicorn" || echo "No uvicorn process found"
 	lsof -ti :8000 | xargs kill -9

# --- Qdrant Seeding ---

# Seed Qdrant with sample data from JSONL file
seed:
	@echo "Seeding Qdrant with sample data..."
	@. venv/bin/activate && python scripts/seed_qdrant.py 2>/dev/null
	@echo "Seeding Gemini collection..."
	@. venv/bin/activate && python scripts/seed_qdrant_gemini.py 2>/dev/null
	@echo " Qdrant seed completed successfully."

# Run OpenAI API smoke test (verifies API key, auth, and connectivity)
# Usage: make smoke-api
smoke-api:
	@echo "Running OpenAI API smoke test..."
	@python3 scripts/api_smoke_test.py


# FOR Development and operations -  Codex model selector (default is set up  "gpt-5o-mini" in  ~/.codex.config.toml

# UX html/css/FASTAPI / ASGI uvicorn 
codex-html: ; codex --model "gpt-5-mini"  --config model_reasoning_effort="low"  
# git vcs, CLI, module refactors, targeted operational tasks 
codex-ops:  ; codex --model "gpt-5-nano"  --config model_reasoning_effort="low"  
# for python coding
codex-code: ; codex --model "gpt-5-mini"   
# for vector db and docker
codex-db:   ; codex --model "gpt-5-mini"  
# for full repo context - expensive model
codex-5:    ; codex --model "gpt-5"       

# My ipaddress on the server to use for connecting to the application from another machine on the same network
# Example Use: http://192.168.86.174:8000 (where the ip address is from the command below

my-ip: ; ipconfig getifaddr en0
