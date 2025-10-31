#!/bin/bash

# Coreference Service Setup Script

echo "🚀 Setting up Coreference Resolution Service..."

# Check Python version
PYTHON_BIN="python3.10"
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "❌ Python 3.10 not found. Install with: brew install python@3.10"
    exit 1
fi
python_version=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_BIN -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "📥 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download coreferee model
echo "📥 Downloading coreferee model..."
python -m coreferee install en

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    
    # Generate random API key
    api_key=$(openssl rand -hex 16)
    
    # Update .env with generated API key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/your-api-key-here/$api_key/" .env
    else
        # Linux
        sed -i "s/your-api-key-here/$api_key/" .env
    fi
    
    echo "✅ Generated API key: $api_key"
    echo "📝 Please update .env with your configuration"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the service:"
echo "  1. source venv/bin/activate"
echo "  2. python server.py"
echo ""
echo "Or simply run: npm start"
