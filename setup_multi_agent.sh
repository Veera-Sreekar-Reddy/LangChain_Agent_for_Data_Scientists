#!/bin/bash

# Setup script for Multi-Agent System
echo "🚀 Setting up Multi-Agent Data Science Assistant..."
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo "Install with: brew install ollama (macOS) or visit https://ollama.com"
    exit 1
fi
echo "✅ Ollama is installed"

# Start Ollama service
echo ""
echo "📡 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 3

# Pull required models
echo ""
echo "📥 Pulling Mistral model (for reasoning)..."
ollama pull mistral
echo "✅ Mistral downloaded"

echo ""
echo "📥 Pulling CodeLLaMA model (for code generation)..."
ollama pull codellama
echo "✅ CodeLLaMA downloaded"

# Verify models
echo ""
echo "🔍 Verifying installed models..."
ollama list

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 You can now run the application with:"
echo "   streamlit run app.py"
echo ""
echo "💡 Tips:"
echo "   • Use 'Multi-Agent Mode' for best results"
echo "   • Mistral handles analysis queries"
echo "   • CodeLLaMA generates visualizations"
echo ""
echo "📚 See ARCHITECTURE.md for detailed documentation"

