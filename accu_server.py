#!/usr/bin/env python3
"""
Accumulator Bot — Deriv
Ativo: Volatility 75 Index (R_75)
Estrategia: Compra ACCU 5%, TP=$0.20, reabre automaticamente ao fechar
"""
import os, json, time, threading, logging
from collections import deque
from datetime import datetime
import websocket
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────────
DERIV_TOKEN   = os.environ.get('DERIV_TOKEN', '')
DERIV_APP_ID  = os.environ.get('DERIV_APP_ID', '1089')
DERIV_WS      = f'wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}'
PORT          = int(os.environ.get('PORT', 8000))

SYMBOL        = 'R_75'       # Volatility 75 Index
STAKE         = 5.0          # USD por ordem
GROWTH_RATE   = 0.05         # 5% crescimento por tick
TAKE_PROFIT   = 0.20         # TP fixo $0.20
REOPEN_DELAY  = 1            # segundos aguardar antes de reabrir
BOT_ACTIVE    = True         # liga/desliga o bot

# ── FLASK ────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── DERIV CLIENT ─────────────────────────────────────────────────────
class DerivClient:
    def __init__(self):
        self._ws        = None
        self._auth      = False
        self._lock      = threading.Lock()
        self._callbacks = {}
        self._req_id    = 0
        self._running   = False

    def _next_id(self):
        with self._lock:
            self._req_id += 1
            return self._req_id

    def send(self, payload: dict, callback=None):
        rid = self._next_id()
        payload['req_id'] = rid
        if callback:
            self._callbacks[rid] = callback
        try:
            self._ws.send(json.dumps(payload))
        except Exception as e:
            log.error(f'send error: {e}')
            if callback:
                self._callbacks.pop(rid, None)
                callback({'error': {'message': str(e)}})

    def _on_message(self, ws, raw):
        try:
            msg  = json.loads(raw)
            rid  = msg.get('req_id')
            mtype = msg.get('msg_type', '')

            # Authorize — marca como autenticado
            if mtype == 'authorize':
                if msg.get('error'):
                    log.error(f'Auth falhou: {msg["error"]["message"]}')
                else:
                    self._auth = True
                    login = msg.get('authorize', {}).get('loginid', '?')
                    log.info(f'Autenticado! Login={login}')

            # Passa para callback registrado
            if rid and rid in self._callbacks:
                # callbacks de proposal ficam ate buy acontecer
                if mtype not in ('proposal',):
                    cb = self._callbacks.pop(rid)
                    cb(msg)
                else:
                    self._callbacks[rid](msg)

            # POC — envia para o trader monitorar
            if mtype == 'proposal_open_contract':
                trader._handle_poc(msg)

        except Exception as e:
            log.error(f'on_message error: {e}')

    def _on_open(self, ws):
        log.info('WS conectado — autenticando...')
        self.send({'authorize': DERIV_TOKEN})

    def _on_error(self, ws, err):
        log.error(f'WS erro: {err}')

    def _on_close(self, ws, code, msg):
        self._auth = False
        log.warning(f'WS fechado ({code}) — reconectando em 5s...')
        time.sleep(5)
        if self._running:
            self._connect()

    def _on_ping(self, ws, data):
        pass

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            DERIV_WS,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        threading.Thread(
            target=self._ws.run_forever,
            kwargs={'ping_interval': 30, 'ping_timeout': 10},
            daemon=True
        ).start()

    def start(self):
        self._running = True
        self._connect()

    @property
    def is_ready(self):
        return self._auth

deriv = DerivClient()

# ── ACCUMULATOR TRADER ───────────────────────────────────────────────
class AccuTrader:
    def __init__(self):
        self._lock       = threading.Lock()
        self._contracts  = {}    # cid → info
        self._opening    = False  # evita abrir duplicado
        self.trades      = deque(maxlen=500)
        self.wins        = 0
        self.losses      = 0
        self.total_pnl   = 0.0
        self.last_error  = None
        self._active     = True

    def start(self):
        """Aguarda WS estar pronto e abre primeira ordem."""
        def _wait():
            for _ in range(30):
                time.sleep(1)
                if deriv._auth:
                    log.info('DerivClient autenticado — iniciando trader')
                    self._open_trade()
                    return
            log.error('Timeout aguardando autenticacao')
        threading.Thread(target=_wait, daemon=True).start()

    def _open_trade(self):
        """Solicita proposta e compra ACCU."""
        global BOT_ACTIVE
        if not BOT_ACTIVE:
            log.info('Bot pausado — nao abre nova ordem')
            return
        if not deriv._auth:
            log.warning('WS nao autenticado — aguardando...')
            time.sleep(3)
            return

        with self._lock:
            if self._opening:
                return
            if len(self._contracts) >= 1:
                log.info(f'Ja tem {len(self._contracts)} ordem aberta — aguardando fechar')
                return
            self._opening = True

        log.info(f'Solicitando proposta ACCU R_75 — stake=${STAKE} growth={int(GROWTH_RATE*100)}% TP=${TAKE_PROFIT}')

        def on_proposal(msg):
            if msg.get('error'):
                err = msg['error']['message']
                log.error(f'Proposta recusada: {err}')
                self.last_error = err
                with self._lock:
                    self._opening = False
                # tenta novamente em 5s
                threading.Timer(5, self._open_trade).start()
                return

            prop = msg.get('proposal', {})
            pid  = prop.get('id')
            if not pid:
                with self._lock:
                    self._opening = False
                return

            log.info(f'Proposta recebida id={pid} — comprando...')
            deriv.send({'buy': pid, 'price': STAKE}, callback=on_buy)

        def on_buy(msg):
            with self._lock:
                self._opening = False

            if msg.get('error'):
                err = msg['error']['message']
                log.error(f'Erro ao comprar: {err}')
                self.last_error = err
                threading.Timer(5, self._open_trade).start()
                return

            buy = msg.get('buy', {})
            cid = buy.get('contract_id')
            if not cid:
                return

            log.info(f'Ordem aberta! contract_id={cid}')
            with self._lock:
                self._contracts[cid] = {
                    'cid':       cid,
                    'open_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'stake':     STAKE,
                }

            # Subscreve para monitorar o contrato
            deriv.send({
                'proposal_open_contract': 1,
                'contract_id': cid,
                'subscribe': 1,
            })

        deriv.send({
            'proposal':     1,
            'amount':       STAKE,
            'basis':        'stake',
            'contract_type':'ACCU',
            'currency':     'USD',
            'growth_rate':  GROWTH_RATE,
            'symbol':       SYMBOL,
            'limit_order':  {'take_profit': TAKE_PROFIT},
        }, callback=on_proposal)

    def _handle_poc(self, msg):
        """Recebe atualizacoes do contrato aberto."""
        poc = msg.get('proposal_open_contract', {})
        if not poc:
            return

        cid    = poc.get('contract_id')
        status = poc.get('status', '')
        is_sold = poc.get('is_sold', 0)

        if not is_sold:
            # Contrato ainda aberto — log do valor atual
            current_pnl = float(poc.get('profit', 0))
            current_val = float(poc.get('current_spot', 0))
            if cid in self._contracts:
                log.debug(f'[{cid}] spot={current_val:.4f} lucro=${current_pnl:.4f}')
            return

        # ─ Contrato fechado ─
        sell_price  = float(poc.get('sell_price', 0))
        buy_price   = float(poc.get('buy_price',  STAKE))
        pnl         = sell_price - buy_price
        sell_time   = poc.get('sell_time', '')
        exit_reason = 'TP' if pnl > 0 else 'LOSS'

        with self._lock:
            info = self._contracts.pop(cid, {})

        if pnl > 0:
            self.wins     += 1
            self.total_pnl += pnl
            status_emoji = 'WIN'
        else:
            self.losses   += 1
            self.total_pnl += pnl
            status_emoji = 'LOSS'

        log.info(
            f'[{status_emoji}] id={cid} PnL=${pnl:+.4f} '
            f'| Total: {self.wins}W/{self.losses}L PnL=${self.total_pnl:+.2f}'
        )

        self.trades.appendleft({
            'cid':       cid,
            'open_time': info.get('open_time', ''),
            'close_time': str(sell_time),
            'stake':      buy_price,
            'pnl':        round(pnl, 4),
            'result':     exit_reason,
            'wins':       self.wins,
            'losses':     self.losses,
        })

        # Reabre automaticamente apos delay
        log.info(f'Reabrindo nova ordem em {REOPEN_DELAY}s...')
        threading.Timer(REOPEN_DELAY, self._open_trade).start()

trader = AccuTrader()

# ── ENDPOINTS ────────────────────────────────────────────────────────
@app.route('/')
def index():
    global BOT_ACTIVE
    winrate = (trader.wins / (trader.wins + trader.losses) * 100
               if (trader.wins + trader.losses) > 0 else 0)
    return jsonify({
        'bot':         'Accumulator R_75',
        'active':      BOT_ACTIVE,
        'symbol':      SYMBOL,
        'stake':       STAKE,
        'growth_rate': f'{int(GROWTH_RATE*100)}%',
        'take_profit': TAKE_PROFIT,
        'ws_ready':    deriv._auth,
        'open_orders': len(trader._contracts),
        'wins':        trader.wins,
        'losses':      trader.losses,
        'winrate':     round(winrate, 1),
        'total_pnl':   round(trader.total_pnl, 2),
        'last_error':  trader.last_error,
    })

@app.route('/trades')
def trades():
    return jsonify(list(trader.trades)[:50])

@app.route('/start')
def start_bot():
    global BOT_ACTIVE
    BOT_ACTIVE = True
    trader._open_trade()
    return jsonify({'ok': True, 'msg': 'Bot iniciado'})

@app.route('/stop')
def stop_bot():
    global BOT_ACTIVE
    BOT_ACTIVE = False
    return jsonify({'ok': True, 'msg': 'Bot pausado — ordens abertas fecham normalmente'})

@app.route('/status')
def status():
    return jsonify({
        'active':      BOT_ACTIVE,
        'ws_ready':    deriv._auth,
        'open_orders': list(trader._contracts.keys()),
        'wins':        trader.wins,
        'losses':      trader.losses,
        'total_pnl':   round(trader.total_pnl, 2),
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# ── INICIALIZA (funciona com gunicorn e python direto) ───────────────
if DERIV_TOKEN:
    deriv.start()
    trader.start()
else:
    log.error('DERIV_TOKEN nao configurado!')

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
