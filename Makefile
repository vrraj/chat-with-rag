# Run FastAPI server in foreground with live reload
run:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# Run FastAPI server in background with logs redirected
bg-run:
	nohup uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug > logs/server.log 2>&1 &

# Tail the background server logs
logs:
	tail -f logs/server.log

# Kill any process running on port 8000
kill:
	lsof -ti :8000 | xargs kill -9

# Start Qdrant in detached (background) mode
qdrant-up:
	docker-compose up -d

# Stop and remove Qdrant container and resources
qdrant-down:
	docker-compose down

# Stream Qdrant logs live
qdrant-logs:
	docker-compose logs -f qdrant

# Restart Qdrant (stop + start)
qdrant-restart: qdrant-down qdrant-up

# Check if Qdrant is running
qdrant-status:
	docker ps | grep qdrant || echo "Qdrant is not running"

# List all Qdrant collections (pretty-printed if jq or Python is available)
qdrant-collections:
	@echo "Listing Qdrant collections from http://localhost:6333/collections"
	@resp=$$(curl -s http://localhost:6333/collections); \
	if command -v jq >/dev/null 2>&1; then \
		echo "$$resp" | jq .; \
	else \
		echo "$$resp" | python3 -m json.tool 2>/dev/null || echo "$$resp"; \
	fi

# Show info for a single collection: make qdrant-info COLLECTION=my_collection
qdrant-info:
	@if [ -z "$(COLLECTION)" ]; then \
		echo "Usage: make qdrant-info COLLECTION=<name>"; \
		exit 1; \
	fi
	@url="http://localhost:6333/collections/$(COLLECTION)"; \
	echo "GET $$url"; \
	resp=$$(curl -s "$$url"); \
	if command -v jq >/dev/null 2>&1; then \
		echo "$$resp" | jq .; \
	else \
		echo "$$resp" | python3 -m json.tool 2>/dev/null || echo "$$resp"; \
	fi

# --- Qdrant storage safety & backups ---

# Backup when using bind mount ./qdrant_storage
qdrant-backup:
	@mkdir -p backups
	@tar -czf backups/qdrant-$(shell date +%F).tar.gz -C ./qdrant_storage . || echo "Nothing to back up yet (./qdrant_storage is empty?)"
	@echo "✅ Backup written to backups/"

# Backup an old/existing named volume (e.g., qdrant_data)
qdrant-backup-volume:
	@if [ -z "$(VOLUME)" ]; then echo "Usage: make qdrant-backup-volume VOLUME=qdrant_data"; exit 1; fi
	@mkdir -p backups
	@docker run --rm -v $(VOLUME):/from -v $(PWD)/backups:/to alpine \
	  sh -lc 'cd /from && tar -czf /to/qdrant-$(VOLUME)-$$(date +%F).tar.gz .' \
	  && echo "✅ Backup written to backups/"

# Migrate data from a named volume into the new bind mount folder
qdrant-migrate-from-volume:
	@if [ -z "$(VOLUME)" ]; then echo "Usage: make qdrant-migrate-from-volume VOLUME=qdrant_data"; exit 1; fi
	@mkdir -p qdrant_storage
	@docker-compose stop qdrant >/dev/null 2>&1 || true
	@docker run --rm -v $(VOLUME):/from -v $(PWD)/qdrant_storage:/to alpine \
	  sh -lc 'cp -a /from/. /to/' \
	  && echo "✅ Migrated data from volume '$(VOLUME)' to ./qdrant_storage"
	@echo "Now run: docker-compose up -d && make qdrant-collections"


.PHONY: kill-uvicorn
kill-uvicorn:
	@echo "Killing any running uvicorn processes..."
	@pkill -f "uvicorn" || echo "No uvicorn process found"

