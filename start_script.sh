#!/bin/bash

# Script de inicialização para Render

# Criar diretórios necessários
mkdir -p models
mkdir -p data
mkdir -p logs

# Verificar se o arquivo principal existe
if [ ! -f "main.py" ]; then
    echo "Erro: main.py não encontrado!"
    exit 1
fi

# Iniciar a aplicação
echo "Iniciando ML Trading API..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info