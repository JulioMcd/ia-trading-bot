<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IQ Option Trading Bot - Professional AI Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-bg: #0a0a0a;
            --secondary-bg: #1a1a1a;
            --card-bg: #262626;
            --accent-color: #00d4aa;
            --accent-hover: #00b294;
            --success-color: #22c55e;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --text-primary: #ffffff;
            --text-secondary: #a1a1aa;
            --border-color: #404040;
            --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            --gradient-primary: linear-gradient(135deg, #00d4aa 0%, #00a085 100%);
            --gradient-card: linear-gradient(135deg, #262626 0%, #1a1a1a 100%);
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--primary-bg);
            color: var(--text-primary);
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            background: var(--gradient-card);
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo i {
            font-size: 2.5rem;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .connection-status {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 12px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            font-size: 0.9rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--danger-color);
            animation: pulse 2s infinite;
        }

        .status-dot.connected {
            background: var(--success-color);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
        }

        /* Cards */
        .card {
            background: var(--gradient-card);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-title i {
            color: var(--accent-color);
        }

        /* Stats Cards */
        .stat-card {
            text-align: center;
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .stat-value.positive {
            color: var(--success-color);
        }

        .stat-value.negative {
            color: var(--danger-color);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Buttons */
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            font-size: 0.9rem;
        }

        .btn-primary {
            background: var(--gradient-primary);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 212, 170, 0.3);
        }

        .btn-danger {
            background: var(--danger-color);
            color: white;
        }

        .btn-secondary {
            background: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-toggle {
            background: var(--border-color);
            color: var(--text-secondary);
        }

        .btn-toggle.active {
            background: var(--gradient-primary);
            color: white;
        }

        /* Forms */
        .form-group {
            margin-bottom: 20px;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .form-input {
            width: 100%;
            padding: 12px 16px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
        }

        .form-select {
            width: 100%;
            padding: 12px 16px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
        }

        /* AI Controls */
        .ai-controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .ai-toggle-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        /* Trade History */
        .trade-history {
            max-height: 400px;
            overflow-y: auto;
        }

        .trade-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .trade-item:last-child {
            border-bottom: none;
        }

        .trade-direction {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
        }

        .trade-direction.call {
            color: var(--success-color);
        }

        .trade-direction.put {
            color: var(--danger-color);
        }

        .trade-result {
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .trade-result.win {
            background: rgba(34, 197, 94, 0.2);
            color: var(--success-color);
        }

        .trade-result.loss {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger-color);
        }

        /* Progress Bars */
        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-fill {
            height: 100%;
            background: var(--gradient-primary);
            transition: width 0.3s ease;
        }

        /* Martingale Display */
        .martingale-info {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid var(--warning-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .martingale-info.active {
            background: rgba(239, 68, 68, 0.1);
            border-color: var(--danger-color);
        }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            backdrop-filter: blur(10px);
        }

        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--gradient-card);
            border-radius: 16px;
            padding: 30px;
            width: 90%;
            max-width: 500px;
            border: 1px solid var(--border-color);
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .header-content {
                text-align: center;
            }
            .ai-toggle-group {
                justify-content: center;
            }
        }

        /* Animations */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .loading {
            position: relative;
            overflow: hidden;
        }

        .loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: loading 1.5s infinite;
        }

        @keyframes loading {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        /* AI Status */
        .ai-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
        }

        .ai-status.connected {
            border-color: var(--success-color);
            background: rgba(34, 197, 94, 0.1);
        }

        .ai-status.error {
            border-color: var(--danger-color);
            background: rgba(239, 68, 68, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header fade-in">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-robot"></i>
                    <h1>IQ Trading Bot AI <span style="font-size: 0.6em; color: var(--accent-color);">Standalone</span></h1>
                </div>
                <div class="connection-status">
                    <div class="status-indicator">
                        <div class="status-dot" id="iqStatus"></div>
                        <span id="iqStatusText">IQ Option</span>
                    </div>
                    <div class="status-indicator">
                        <div class="status-dot" id="aiStatus"></div>
                        <span id="aiStatusText">AI Engine</span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Dashboard Stats -->
        <div class="dashboard-grid fade-in">
            <div class="card stat-card">
                <div class="card-title">
                    <i class="fas fa-chart-line"></i>
                    P&L Sessão
                </div>
                <div class="stat-value" id="sessionPnL">$0.00</div>
                <div class="stat-label">Profit & Loss</div>
            </div>

            <div class="card stat-card">
                <div class="card-title">
                    <i class="fas fa-percentage"></i>
                    Win Rate
                </div>
                <div class="stat-value" id="winRate">0%</div>
                <div class="stat-label">Taxa de Acerto</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="winRateProgress" style="width: 0%"></div>
                </div>
            </div>

            <div class="card stat-card">
                <div class="card-title">
                    <i class="fas fa-exchange-alt"></i>
                    Trades
                </div>
                <div class="stat-value" id="tradesCount">0</div>
                <div class="stat-label"><span id="wins">0</span>W / <span id="losses">0</span>L</div>
            </div>

            <div class="card stat-card">
                <div class="card-title">
                    <i class="fas fa-wallet"></i>
                    Saldo
                </div>
                <div class="stat-value" id="currentBalance">$0.00</div>
                <div class="stat-label">Conta Demo</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-grid">
            <!-- Left Column -->
            <div>
                <!-- Login Card -->
                <div class="card fade-in" id="loginCard">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fas fa-sign-in-alt"></i>
                            Conectar IQ Option
                        </div>
                    </div>
                    <form id="loginForm">
                        <div class="form-group">
                            <label class="form-label">Email</label>
                            <input type="email" class="form-input" id="email" placeholder="demo@exemplo.com (qualquer email)" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Senha</label>
                            <input type="password" class="form-input" id="password" placeholder="••••••••• (qualquer senha)" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Tipo de Conta</label>
                            <select class="form-select" id="accountType">
                                <option value="PRACTICE">Demo (Recomendado)</option>
                                <option value="REAL">Real</option>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-primary" style="width: 100%;">
                            <i class="fas fa-plug"></i>
                            Conectar
                        </button>
                    </form>
                </div>

                <!-- Bot Controls -->
                <div class="card fade-in" id="botControls" style="display: none;">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fas fa-robot"></i>
                            Controles do Bot
                        </div>
                        <div id="botStatus" class="status-indicator">
                            <div class="status-dot"></div>
                            <span>Parado</span>
                        </div>
                    </div>

                    <!-- AI Status -->
                    <div class="ai-status" id="aiStatusCard">
                        <i class="fas fa-brain"></i>
                        <span id="aiStatusMessage">Conectando à IA...</span>
                    </div>

                    <!-- Martingale Info -->
                    <div class="martingale-info" id="martingaleInfo" style="display: none;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>Martingale Ativo</strong>
                                <div style="font-size: 0.9rem; margin-top: 5px;">
                                    Nível <span id="martingaleLevel">0</span> | Próximo: $<span id="nextStake">0</span>
                                </div>
                            </div>
                            <button class="btn btn-secondary" onclick="resetMartingale()">
                                <i class="fas fa-undo"></i>
                                Reset
                            </button>
                        </div>
                    </div>

                    <!-- Bot Buttons -->
                    <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                        <button class="btn btn-primary" id="startBot" onclick="startBot()" style="flex: 1;">
                            <i class="fas fa-play"></i>
                            Iniciar Bot
                        </button>
                        <button class="btn btn-danger" id="stopBot" onclick="stopBot()" style="flex: 1; display: none;">
                            <i class="fas fa-stop"></i>
                            Parar Bot
                        </button>
                    </div>

                    <!-- Manual Trade -->
                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-success" onclick="manualTrade('call')" style="flex: 1;">
                            <i class="fas fa-arrow-up"></i>
                            CALL
                        </button>
                        <button class="btn btn-danger" onclick="manualTrade('put')" style="flex: 1;">
                            <i class="fas fa-arrow-down"></i>
                            PUT
                        </button>
                    </div>
                </div>

                <!-- Trade History -->
                <div class="card fade-in">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fas fa-history"></i>
                            Histórico de Trades
                        </div>
                        <button class="btn btn-secondary" onclick="loadHistory()">
                            <i class="fas fa-refresh"></i>
                        </button>
                    </div>
                    <div class="trade-history" id="tradeHistory">
                        <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                            <i class="fas fa-chart-line" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.3;"></i>
                            <div>Nenhum trade realizado ainda</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div>
                <!-- AI Controls -->
                <div class="card fade-in">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fas fa-brain"></i>
                            Controles de IA
                        </div>
                    </div>
                    <div class="ai-controls">
                        <div class="ai-toggle-group">
                            <button class="btn btn-toggle" id="aiModeBtn" onclick="toggleAIMode('trading')">
                                <i class="fas fa-robot"></i>
                                IA Trading
                            </button>
                            <button class="btn btn-toggle" id="aiDurationBtn" onclick="toggleAIMode('duration')">
                                <i class="fas fa-clock"></i>
                                IA Duração
                            </button>
                            <button class="btn btn-toggle" id="aiManagementBtn" onclick="toggleAIMode('management')">
                                <i class="fas fa-cog"></i>
                                IA Gestão
                            </button>
                        </div>
                        
                        <div class="form-group">
                            <button class="btn btn-primary" onclick="getAIAnalysis()" style="width: 100%;">
                                <i class="fas fa-chart-area"></i>
                                Análise IA
                            </button>
                        </div>

                        <div class="form-group">
                            <button class="btn btn-primary" onclick="getAISignal()" style="width: 100%;">
                                <i class="fas fa-crosshairs"></i>
                                Sinal IA
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Bot Configuration -->
                <div class="card fade-in">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fas fa-cog"></i>
                            Configurações
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Ativo</label>
                        <select class="form-select" id="symbol">
                            <option value="EURUSD-OTC">EURUSD-OTC</option>
                            <option value="GBPUSD-OTC">GBPUSD-OTC</option>
                            <option value="USDJPY-OTC">USDJPY-OTC</option>
                            <option value="AUDUSD-OTC">AUDUSD-OTC</option>
                            <option value="USDCAD-OTC">USDCAD-OTC</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Valor Base ($)</label>
                        <input type="number" class="form-input" id="baseAmount" value="1" min="1" max="1000">
                    </div>

                    <div class="form-group">
                        <label class="form-label">Duração (min)</label>
                        <input type="number" class="form-input" id="duration" value="1" min="1" max="5">
                    </div>

                    <div class="form-group">
                        <label class="form-label">Multiplicador Martingale</label>
                        <input type="number" class="form-input" id="martingaleMultiplier" value="2.2" step="0.1" min="1.1" max="5">
                    </div>

                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-toggle" id="martingaleBtn" onclick="toggleMartingale()">
                            <i class="fas fa-dice"></i>
                            Martingale
                        </button>
                        <button class="btn btn-secondary" onclick="saveConfig()" style="flex: 1;">
                            <i class="fas fa-save"></i>
                            Salvar
                        </button>
                    </div>
                </div>

                <!-- AI Insights -->
                <div class="card fade-in">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fas fa-lightbulb"></i>
                            Insights da IA
                        </div>
                    </div>
                    <div id="aiInsights" style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;">
                        <div style="text-align: center; padding: 20px;">
                            <i class="fas fa-brain" style="font-size: 2rem; margin-bottom: 10px; opacity: 0.3;"></i>
                            <div>Clique em "Análise IA" para insights</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Modal -->
    <div class="modal" id="loadingModal">
        <div class="modal-content" style="text-align: center;">
            <i class="fas fa-spinner fa-spin" style="font-size: 3rem; color: var(--accent-color); margin-bottom: 20px;"></i>
            <h3 id="loadingText">Processando...</h3>
            <p style="color: var(--text-secondary); margin-top: 10px;" id="loadingSubtext">Aguarde...</p>
        </div>
    </div>

    <script>
        // Configuration
        const CONFIG = {
            AI_API_URL: 'https://ia-trading-bot-nrn1.onrender.com', // URL da sua IA
            BOT_API_URL: 'http://localhost:5000', // Backend Python local
            UPDATE_INTERVAL: 2000, // 2 segundos
            AI_RETRY_INTERVAL: 5000 // 5 segundos
        };

        // Global State
        let botState = {
            connected: false,
            running: false,
            aiConnected: false,
            aiModes: {
                trading: false,
                duration: false,
                management: false
            }
        };

        let statsData = {
            balance: 0,
            sessionPnL: 0,
            trades: 0,
            wins: 0,
            losses: 0,
            winRate: 0
        };

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 IQ Trading Bot AI Dashboard Iniciado');
            checkAIConnection();
            updateUI();
            setInterval(updateStats, CONFIG.UPDATE_INTERVAL);
            setInterval(checkAIConnection, CONFIG.AI_RETRY_INTERVAL);
        });

        // API Functions
        async function apiCall(url, method = 'GET', data = null) {
            try {
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                    }
                };

                if (data) {
                    options.body = JSON.stringify(data);
                }

                const response = await fetch(url, options);
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                return { success: false, error: error.message };
            }
        }

        // AI Connection
        async function checkAIConnection() {
            try {
                const response = await fetch(CONFIG.AI_API_URL + '/health');
                if (response.ok) {
                    botState.aiConnected = true;
                    updateAIStatus('connected', 'IA Conectada');
                } else {
                    throw new Error('AI not responding');
                }
            } catch (error) {
                botState.aiConnected = false;
                updateAIStatus('error', 'IA Desconectada');
            }
        }

        function updateAIStatus(status, message) {
            const statusCard = document.getElementById('aiStatusCard');
            const statusMessage = document.getElementById('aiStatusMessage');
            const aiStatusDot = document.getElementById('aiStatus');
            const aiStatusText = document.getElementById('aiStatusText');

            statusCard.className = `ai-status ${status}`;
            statusMessage.textContent = message;
            
            if (status === 'connected') {
                aiStatusDot.classList.add('connected');
                aiStatusText.textContent = 'AI Online';
            } else {
                aiStatusDot.classList.remove('connected');
                aiStatusText.textContent = 'AI Offline';
            }
        }

        // Login Functions
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            showLoading('Conectando...', 'Aguardando resposta da IQ Option');
            
            const loginData = {
                email: document.getElementById('email').value,
                password: document.getElementById('password').value,
                account_type: document.getElementById('accountType').value
            };

            try {
                const result = await apiCall(CONFIG.BOT_API_URL + '/api/login', 'POST', loginData);
                
                hideLoading();

                if (result.success) {
                    botState.connected = true;
                    statsData.balance = result.balance;
                    document.getElementById('loginCard').style.display = 'none';
                    document.getElementById('botControls').style.display = 'block';
                    updateConnectionStatus();
                    updateStatsDisplay();
                    showNotification(`Conectado com sucesso! Saldo: ${result.balance}`, 'success');
                } else {
                    showNotification('Erro no login: ' + result.error, 'error');
                }
            } catch (error) {
                hideLoading();
                showNotification('Erro de conexão: Certifique-se que o bot Python está rodando na porta 5000', 'error');
                console.error('Erro de conexão:', error);
            }
        });

        // Bot Controls
        async function startBot() {
            if (!botState.connected) {
                showNotification('Conecte-se primeiro à IQ Option', 'warning');
                return;
            }

            showLoading('Iniciando Bot...', 'Configurando IA e parâmetros (Demo)');

            // Simular inicialização do bot
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            hideLoading();

            botState.running = true;
            updateBotStatus();
            showNotification('Bot iniciado com sucesso! (Modo Demo)', 'success');
            
            // Iniciar simulação de trading automático
            if (botState.running) {
                setTimeout(simulateAutoTrade, 5000); // Primeiro trade em 5s
            }
        }

        async function stopBot() {
            showLoading('Parando Bot...', 'Finalizando operações');

            await new Promise(resolve => setTimeout(resolve, 1000));
            
            hideLoading();

            botState.running = false;
            updateBotStatus();
            showNotification('Bot parado com sucesso!', 'success');
        }

        async function manualTrade(direction) {
            if (!botState.connected) {
                showNotification('Conecte-se primeiro à IQ Option', 'warning');
                return;
            }

            showLoading(`Executando ${direction.toUpperCase()}...`, 'Processando trade manual (Demo)');

            const tradeData = {
                direction: direction,
                amount: parseFloat(document.getElementById('baseAmount').value),
                duration: parseInt(document.getElementById('duration').value)
            };

            // Simular execução do trade
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            hideLoading();

            showNotification(`Trade ${direction.toUpperCase()} executado! (Demo)`, 'success');
            
            // Simular resultado do trade após a duração
            setTimeout(() => {
                simulateTradeResult(tradeData);
            }, tradeData.duration * 1000); // Simular duração em segundos em vez de minutos para demo
        }

        // AI Functions
        async function toggleAIMode(mode) {
            try {
                const result = await apiCall(CONFIG.BOT_API_URL + `/api/ai/toggle/${mode}`, 'POST');
                
                if (result.success) {
                    botState.aiModes[mode] = result.active;
                    updateAIModeButtons();
                    showNotification(`IA ${mode} ${result.active ? 'ativada' : 'desativada'}`, 'info');
                } else {
                    showNotification('Erro ao alterar modo IA', 'error');
                }
            } catch (error) {
                showNotification('Erro de conexão com o bot', 'error');
            }
        }

        async function getAIAnalysis() {
            if (!botState.aiConnected) {
                showNotification('IA não conectada', 'warning');
                return;
            }

            showLoading('Analisando Mercado...', 'IA processando dados');

            try {
                const analysisData = {
                    symbol: document.getElementById('symbol').value,
                    balance: statsData.balance,
                    win_rate: statsData.winRate
                };

                const result = await apiCall(CONFIG.AI_API_URL + '/analyze', 'POST', analysisData);
                
                hideLoading();

                if (result.status === 'success') {
                    displayAIInsights(result);
                    showNotification('Análise IA concluída!', 'success');
                } else {
                    showNotification('Erro na análise IA', 'error');
                }
            } catch (error) {
                hideLoading();
                showNotification('Erro ao conectar com IA', 'error');
            }
        }

        async function getAISignal() {
            if (!botState.aiConnected) {
                showNotification('IA não conectada', 'warning');
                return;
            }

            showLoading('Gerando Sinal...', 'IA analisando padrões');

            try {
                const signalData = {
                    symbol: document.getElementById('symbol').value,
                    balance: statsData.balance,
                    win_rate: statsData.winRate,
                    martingale_level: 0
                };

                const result = await apiCall(CONFIG.AI_API_URL + '/signal', 'POST', signalData);
                
                hideLoading();

                if (result.status === 'success') {
                    displayAISignal(result);
                    showNotification(`Sinal IA: ${result.direction.toUpperCase()} - ${result.confidence.toFixed(1)}%`, 'info');
                } else {
                    showNotification('Erro no sinal IA', 'error');
                }
            } catch (error) {
                hideLoading();
                showNotification('Erro ao conectar com IA', 'error');
            }
        }

        function displayAIInsights(analysis) {
            const insights = document.getElementById('aiInsights');
            insights.innerHTML = `
                <div style="margin-bottom: 15px;">
                    <strong style="color: var(--accent-color);">📊 Análise de Mercado</strong>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Símbolo:</strong> ${analysis.symbol}
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Confiança:</strong> <span style="color: var(--success-color);">${analysis.confidence.toFixed(1)}%</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Volatilidade:</strong> ${analysis.volatility.toFixed(1)}%
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Tendência:</strong> ${analysis.trend}
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>Condição:</strong> ${analysis.analysis.market_condition}
                </div>
                <div style="padding: 10px; background: rgba(0, 212, 170, 0.1); border-radius: 8px; border-left: 3px solid var(--accent-color);">
                    ${analysis.message}
                </div>
            `;
        }

        function displayAISignal(signal) {
            const insights = document.getElementById('aiInsights');
            const directionColor = signal.direction === 'call' ? 'var(--success-color)' : 'var(--danger-color)';
            const directionIcon = signal.direction === 'call' ? '📈' : '📉';
            
            insights.innerHTML = `
                <div style="margin-bottom: 15px;">
                    <strong style="color: var(--accent-color);">${directionIcon} Sinal IA</strong>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Direção:</strong> <span style="color: ${directionColor}; font-weight: bold;">${signal.direction.toUpperCase()}</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Confiança:</strong> <span style="color: var(--success-color);">${signal.confidence.toFixed(1)}%</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Volatilidade:</strong> ${signal.volatility.toFixed(1)}%
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Condição:</strong> ${signal.market_condition}
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>Timeframe:</strong> ${signal.optimal_timeframe.duration}${signal.optimal_timeframe.type === 'minutes' ? 'm' : 't'}
                </div>
                <div style="padding: 10px; background: rgba(0, 212, 170, 0.1); border-radius: 8px; border-left: 3px solid var(--accent-color);">
                    <strong>Raciocínio:</strong> ${signal.reasoning}
                </div>
            `;
        }

        // Update Functions
        async function updateStats() {
            if (!botState.connected) return;

            try {
                const result = await apiCall(CONFIG.BOT_API_URL + '/api/stats');
                
                if (result.success) {
                    const stats = result.session_stats;
                    statsData = {
                        balance: stats.current_balance,
                        sessionPnL: stats.profit_loss,
                        trades: stats.trades_count,
                        wins: stats.wins,
                        losses: stats.losses,
                        winRate: stats.win_rate
                    };

                    updateStatsDisplay();

                    // Update martingale info
                    if (result.martingale && result.martingale.active) {
                        showMartingaleInfo(result.martingale);
                    } else {
                        hideMartingaleInfo();
                    }

                    // Update bot running status
                    if (result.bot_running !== botState.running) {
                        botState.running = result.bot_running;
                        updateBotStatus();
                    }
                }
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }

        function updateStatsDisplay() {
            document.getElementById('currentBalance').textContent = `$${statsData.balance.toFixed(2)}`;
            
            const pnlElement = document.getElementById('sessionPnL');
            pnlElement.textContent = `$${statsData.sessionPnL.toFixed(2)}`;
            pnlElement.className = `stat-value ${statsData.sessionPnL >= 0 ? 'positive' : 'negative'}`;
            
            document.getElementById('winRate').textContent = `${statsData.winRate.toFixed(1)}%`;
            document.getElementById('winRateProgress').style.width = `${statsData.winRate}%`;
            
            document.getElementById('tradesCount').textContent = statsData.trades;
            document.getElementById('wins').textContent = statsData.wins;
            document.getElementById('losses').textContent = statsData.losses;
        }

        function updateConnectionStatus() {
            const iqStatus = document.getElementById('iqStatus');
            const iqStatusText = document.getElementById('iqStatusText');
            
            if (botState.connected) {
                iqStatus.classList.add('connected');
                iqStatusText.textContent = 'IQ Conectada';
            } else {
                iqStatus.classList.remove('connected');
                iqStatusText.textContent = 'IQ Desconectada';
            }
        }

        function updateBotStatus() {
            const startBtn = document.getElementById('startBot');
            const stopBtn = document.getElementById('stopBot');
            const statusIndicator = document.querySelector('#botStatus .status-dot');
            const statusText = document.querySelector('#botStatus span');
            
            if (botState.running) {
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-flex';
                statusIndicator.classList.add('connected');
                statusText.textContent = 'Rodando';
            } else {
                startBtn.style.display = 'inline-flex';
                stopBtn.style.display = 'none';
                statusIndicator.classList.remove('connected');
                statusText.textContent = 'Parado';
            }
        }

        function updateAIModeButtons() {
            const buttons = {
                trading: document.getElementById('aiModeBtn'),
                duration: document.getElementById('aiDurationBtn'),
                management: document.getElementById('aiManagementBtn')
            };

            Object.keys(buttons).forEach(mode => {
                const btn = buttons[mode];
                if (botState.aiModes[mode]) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        function showMartingaleInfo(martingale) {
            const info = document.getElementById('martingaleInfo');
            document.getElementById('martingaleLevel').textContent = martingale.level;
            document.getElementById('nextStake').textContent = martingale.next_amount.toFixed(2);
            
            if (martingale.level > 4) {
                info.classList.add('active');
            }
            
            info.style.display = 'block';
        }

        function hideMartingaleInfo() {
            document.getElementById('martingaleInfo').style.display = 'none';
        }

        // Utility Functions
        function showLoading(title, subtitle) {
            document.getElementById('loadingText').textContent = title;
            document.getElementById('loadingSubtext').textContent = subtitle;
            document.getElementById('loadingModal').style.display = 'block';
        }

        function hideLoading() {
            document.getElementById('loadingModal').style.display = 'none';
        }

        function showNotification(message, type) {
            // Simple notification system
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--gradient-card);
                color: var(--text-primary);
                padding: 15px 20px;
                border-radius: 12px;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow);
                z-index: 10000;
                max-width: 300px;
                animation: slideIn 0.3s ease;
            `;
            
            const colors = {
                success: 'var(--success-color)',
                error: 'var(--danger-color)',
                warning: 'var(--warning-color)',
                info: 'var(--accent-color)'
            };
            
            notification.style.borderColor = colors[type] || colors.info;
            notification.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px;">
                    <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation-triangle' : 'info'}"></i>
                    <span>${message}</span>
                </div>
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }

        function updateUI() {
            updateConnectionStatus();
            updateBotStatus();
            updateAIModeButtons();
            updateStatsDisplay();
        }

        // Configuration Functions
        async function saveConfig() {
            // Simular salvamento de configuração
            showNotification('Configurações salvas! (Demo)', 'success');
        }

        async function toggleMartingale() {
            const btn = document.getElementById('martingaleBtn');
            btn.classList.toggle('active');
            const enabled = btn.classList.contains('active');
            showNotification(`Martingale ${enabled ? 'ativado' : 'desativado'}`, 'info');
        }

        async function resetMartingale() {
            hideMartingaleInfo();
            showNotification('Martingale resetado!', 'success');
        }

        async function loadHistory() {
            showNotification('Histórico carregado!', 'info');
        }

        // Demo Simulation Functions
        function simulateTradeResult(tradeData) {
            // Simular resultado baseado em probabilidade
            const winProbability = 0.6; // 60% de chance de vitória
            const isWin = Math.random() < winProbability;
            
            const payout = 0.8; // 80% de retorno
            let profit;
            
            if (isWin) {
                profit = tradeData.amount * payout;
                statsData.wins++;
                showNotification(`🏆 VITÓRIA! +${profit.toFixed(2)}`, 'success');
            } else {
                profit = -tradeData.amount;
                statsData.losses++;
                showNotification(`💥 DERROTA! -${tradeData.amount.toFixed(2)}`, 'error');
            }
            
            // Atualizar estatísticas
            statsData.trades++;
            statsData.sessionPnL += profit;
            statsData.balance += profit;
            statsData.winRate = (statsData.wins / statsData.trades) * 100;
            
            updateStatsDisplay();
            
            // Adicionar ao histórico visual
            addTradeToHistory({
                direction: tradeData.direction,
                amount: tradeData.amount,
                result: isWin ? 'win' : 'loss',
                profit: profit,
                timestamp: new Date()
            });
        }

        function simulateAutoTrade() {
            if (!botState.running) return;
            
            // Simular trade automático a cada 30-60 segundos
            const directions = ['call', 'put'];
            const direction = directions[Math.floor(Math.random() * directions.length)];
            const amount = parseFloat(document.getElementById('baseAmount').value);
            const duration = parseInt(document.getElementById('duration').value);
            
            showNotification(`🤖 Bot executou: ${direction.toUpperCase()} ${amount}`, 'info');
            
            // Simular resultado após alguns segundos
            setTimeout(() => {
                simulateTradeResult({ direction, amount, duration });
            }, duration * 1000);
            
            // Agendar próximo trade
            if (botState.running) {
                const nextTradeDelay = Math.random() * 30000 + 30000; // 30-60 segundos
                setTimeout(simulateAutoTrade, nextTradeDelay);
            }
        }

        function simulateHistoricalTrades() {
            const trades = [
                { direction: 'call', amount: 1, result: 'win', profit: 0.8, timestamp: new Date(Date.now() - 300000) },
                { direction: 'put', amount: 1, result: 'loss', profit: -1, timestamp: new Date(Date.now() - 240000) },
                { direction: 'call', amount: 2.2, result: 'win', profit: 1.76, timestamp: new Date(Date.now() - 180000) },
                { direction: 'put', amount: 1, result: 'win', profit: 0.8, timestamp: new Date(Date.now() - 120000) },
                { direction: 'call', amount: 1, result: 'loss', profit: -1, timestamp: new Date(Date.now() - 60000) }
            ];
            
            trades.forEach(trade => {
                addTradeToHistory(trade);
                if (trade.result === 'win') {
                    statsData.wins++;
                } else {
                    statsData.losses++;
                }
                statsData.trades++;
                statsData.sessionPnL += trade.profit;
            });
            
            statsData.winRate = (statsData.wins / statsData.trades) * 100;
            updateStatsDisplay();
        }

        function addTradeToHistory(trade) {
            const historyContainer = document.getElementById('tradeHistory');
            
            // Limpar mensagem de "nenhum trade" se existir
            if (historyContainer.children.length === 1 && historyContainer.children[0].style.textAlign === 'center') {
                historyContainer.innerHTML = '';
            }
            
            const tradeElement = document.createElement('div');
            tradeElement.className = 'trade-item fade-in';
            tradeElement.innerHTML = `
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="trade-direction ${trade.direction}">
                        <i class="fas fa-arrow-${trade.direction === 'call' ? 'up' : 'down'}"></i>
                        ${trade.direction.toUpperCase()}
                    </div>
                    <div style="color: var(--text-secondary);">
                        ${trade.amount.toFixed(2)}
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="trade-result ${trade.result}">
                        ${trade.result.toUpperCase()}
                    </div>
                    <div style="font-weight: 600; color: ${trade.profit >= 0 ? 'var(--success-color)' : 'var(--danger-color)'};">
                        ${trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                    </div>
                </div>
            `;
            
            // Adicionar no topo
            historyContainer.insertBefore(tradeElement, historyContainer.firstChild);
            
            // Manter apenas os últimos 10 trades
            while (historyContainer.children.length > 10) {
                historyContainer.removeChild(historyContainer.lastChild);
            }
        }

        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);

        console.log('🎯 Sistema carregado e pronto para uso! (Modo Standalone + IA)');
    </script>
</body>
</html>
