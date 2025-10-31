.PHONY: help install setup migrate run test test-verbose test-headed test-debug clean reset playwright-install

# Default target
help:
	@echo "========================================="
	@echo "HTMX + Alpine.js Pattern Examples"
	@echo "========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          - Create venv and install all dependencies"
	@echo "  make setup            - Full setup (install + migrate)"
	@echo "  make playwright-install - Install Playwright browsers"
	@echo ""
	@echo "Development Commands:"
	@echo "  make migrate          - Run Django migrations"
	@echo "  make run              - Start development server"
	@echo "  make shell            - Open Django shell"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test             - Run all Playwright tests (headless)"
	@echo "  make test-verbose     - Run tests with verbose output"
	@echo "  make test-headed      - Run tests with visible browser"
	@echo "  make test-debug       - Run tests in debug mode (pause on failure)"
	@echo "  make test-dropdowns   - Test only dependent dropdowns"
	@echo "  make test-visibility  - Test only conditional visibility"
	@echo "  make test-state       - Test only state restoration"
	@echo "  make test-report      - Generate HTML test report"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean            - Remove Python cache and test artifacts"
	@echo "  make reset            - Clean + delete venv and database"
	@echo ""

# Installation and setup
install:
	@echo "📦 Creating virtual environment..."
	uv venv
	@echo "📦 Installing Python dependencies..."
	. .venv/bin/activate && uv pip install -r requirements.txt
	@echo "✓ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. source .venv/bin/activate"
	@echo "  2. make playwright-install"
	@echo "  3. make migrate"
	@echo "  4. make test"

playwright-install:
	@echo "🎭 Installing Playwright browsers..."
	. .venv/bin/activate && playwright install chromium
	@echo "✓ Playwright browsers installed!"

setup: install migrate playwright-install
	@echo ""
	@echo "✓ Complete setup finished!"
	@echo ""
	@echo "Run 'make test' to verify patterns work correctly"

# Django commands
migrate:
	@echo "🗄️  Running migrations..."
	. .venv/bin/activate && python manage.py migrate
	@echo "✓ Database ready!"

run:
	@echo "🚀 Starting development server..."
	@echo "   Open: http://localhost:8000/patterns/"
	. .venv/bin/activate && python manage.py runserver

shell:
	. .venv/bin/activate && python manage.py shell

# Testing commands
test:
	@echo "🧪 Running Playwright tests (headless)..."
	. .venv/bin/activate && pytest tests/ -v --tb=short

test-verbose:
	@echo "🧪 Running tests with verbose output..."
	. .venv/bin/activate && pytest tests/ -vv --tb=long -s

test-headed:
	@echo "🧪 Running tests with visible browser..."
	. .venv/bin/activate && pytest tests/ -v --headed --slowmo=500

test-debug:
	@echo "🐛 Running tests in debug mode..."
	. .venv/bin/activate && PWDEBUG=1 pytest tests/ -v -s

test-dropdowns:
	@echo "🧪 Testing dependent dropdowns pattern..."
	. .venv/bin/activate && pytest tests/test_dependent_dropdowns.py -v

test-visibility:
	@echo "🧪 Testing conditional visibility pattern..."
	. .venv/bin/activate && pytest tests/test_conditional_visibility.py -v

test-state:
	@echo "🧪 Testing state restoration pattern..."
	. .venv/bin/activate && pytest tests/test_state_restoration.py -v

test-report:
	@echo "📊 Generating HTML test report..."
	. .venv/bin/activate && pytest tests/ -v --html=test-report.html --self-contained-html
	@echo "✓ Report saved to: test-report.html"

# Maintenance
clean:
	@echo "🧹 Cleaning Python cache and test artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf test-results/
	rm -f test-report.html
	@echo "✓ Cleaned!"

reset: clean
	@echo "⚠️  Resetting environment (deleting venv and database)..."
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf .venv; \
		rm -f db.sqlite3; \
		echo "✓ Reset complete! Run 'make setup' to start fresh."; \
	else \
		echo "Cancelled."; \
	fi

# CI/CD targets
ci-test:
	@echo "🤖 Running CI tests..."
	pytest tests/ -v --tb=short --maxfail=1

# Quick verification (run before committing)
verify: test
	@echo ""
	@echo "✅ All patterns verified!"
	@echo ""
	@echo "Safe to proceed with full platform development."
