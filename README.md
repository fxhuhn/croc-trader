# croc-trader

Python Flask trading-related web application

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:5000`

### Docker

```bash
# Build the image
docker build -t croc-trader .

# Run the container
docker run -p 5000:5000 croc-trader
```

## API Endpoints

- `GET /` - Returns app status and version
- `GET /health` - Health check endpoint

## Development

This is a minimal Flask application structure ready for building trading-related features.
