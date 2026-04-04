#!/usr/bin/env python3
"""
Accumulator Bot — Deriv
Abre ACCU simultaneamente em TODOS os ativos disponiveis.
Ao fechar qualquer ordem, reabre no mesmo ativo automaticamente.
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

STAKE_BASE    = 1.0    # USD stake inicial
STAKE_MAX     = 8.0    # USD stake maximo (3 doublings: 1→2→4→8)
GROWTH_RATE   = 0.05   # 5% crescimento por tick
TAKE_PROFIT   = 0.80   # TP fixo $0.80
REOPEN_DELAY  = 1      # segundos antes de reabrir
BOT_ACTIVE    = True

# Todos os ativos que suportam Accumulators na Deriv
SYMBOLS = [
    'R_10',    # Volatility 10 Index
    'R_25',    # Volatility 25 Index
    'R_50',    # Volatility 50 Index
    'R_75',    # Volatility 75 Index
    'R_100',   # Volatility 100 Index
    '1HZ10V',  # Volatility 10 (1s) Index
    '1HZ25V',  # Volatility 25 (1s) Index
    '1HZ50V',  # Volatility 50 (1s) Index
    '1HZ75V',  # Volatility 75 (1s) Index
    '1HZ100V', # Volatility 100 (1s) Index
]

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
            msg   = json.loads(raw)
            rid   = msg.get('req_id')
            mtype = msg.get('msg_type', '')

            # Authorize
            if mtype == 'authorize':
                if msg.get('error'):
                    log.error(f'Auth falhou: {msg["error"]["message"]}')
                else:
                    self._auth = True
                    login = msg.get('authorize', {}).get('loginid', '?')
                    log.info(f'Autenticado! Login={login}')

            # Callback registrado
            if rid and rid in self._callbacks:
                if mtype not in ('proposal',):
                    cb = self._callbacks.pop(rid)
                    cb(msg)
                else:
                    self._callbacks[rid](msg)

            # POC — repassa ao trader
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
        self._lock          = threading.Lock()
        self._contracts     = {}    # cid → {symbol, open_time, stake}
        self._opening       = set() # symbols com abertura em andamento
        self.trades         = deque(maxlen=1000)
        self.wins           = 0
        self.losses         = 0
        self.total_pnl      = 0.0
        self.last_error     = None
        self.sym_stats      = {s: {'wins': 0, 'losses': 0, 'pnl': 0.0} for s in SYMBOLS}
        self.disabled_syms  = set()
        # Martingale: stake atual por ativo
        self.sym_stake      = {s: STAKE_BASE for s in SYMBOLS}

    def start(self):
        def _wait():
            for _ in range(60):
                time.sleep(1)
                if deriv._auth:
                    log.info(f'Autenticado — abrindo ordens em {len(SYMBOLS)} ativos...')
                    # Abre uma ordem em cada ativo com delay entre elas
                    for i, sym in enumerate(SYMBOLS):
                        threading.Timer(i * 0.5, self._open_trade, args=(sym,)).start()
                    return
            log.error('Timeout aguardando autenticacao')
        threading.Thread(target=_wait, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _symbols_open(self):
        """Retorna set de symbols com contrato aberto."""
        return {info['symbol'] for info in self._contracts.values()}

    def _open_trade(self, symbol: str):
        global BOT_ACTIVE
        if not BOT_ACTIVE:
            return
        if not deriv._auth:
            threading.Timer(5, self._open_trade, args=(symbol,)).start()
            return

        with self._lock:
            if symbol in self.disabled_syms:
                return
            if symbol in self._opening:
                return
            if symbol in self._symbols_open():
                return
            self._opening.add(symbol)

        # Usa stake salvo pelo Martingale (_handle_poc é a fonte de verdade)
        stake_to_use = self.sym_stake.get(symbol, STAKE_BASE)
        martingale_active = stake_to_use > STAKE_BASE
        sym_pnl = self.sym_stats.get(symbol, {}).get('pnl', 0.0)

        log.info(f'[{symbol}] Abrindo ACCU stake=${stake_to_use:.2f} '
                 f'growth={int(GROWTH_RATE*100)}% TP=${TAKE_PROFIT} '
                 f'pnl=${sym_pnl:.2f} martingale={"ON" if martingale_active else "OFF"}')

        def on_proposal(msg):
            if msg.get('error'):
                err = msg['error']['message']
                log.error(f'[{symbol}] Proposta recusada: {err}')
                self.last_error = f'{symbol}: {err}'
                with self._lock:
                    self._opening.discard(symbol)
                threading.Timer(10, self._open_trade, args=(symbol,)).start()
                return

            prop = msg.get('proposal', {})
            pid  = prop.get('id')
            if not pid:
                with self._lock:
                    self._opening.discard(symbol)
                return

            deriv.send({'buy': pid, 'price': stake_to_use}, callback=on_buy)

        def on_buy(msg):
            with self._lock:
                self._opening.discard(symbol)

            if msg.get('error'):
                err = msg['error']['message']
                log.error(f'[{symbol}] Erro ao comprar: {err}')
                self.last_error = f'{symbol}: {err}'
                threading.Timer(10, self._open_trade, args=(symbol,)).start()
                return

            buy = msg.get('buy', {})
            cid = buy.get('contract_id')
            if not cid:
                threading.Timer(5, self._open_trade, args=(symbol,)).start()
                return

            log.info(f'[{symbol}] Ordem aberta! cid={cid} stake=${stake_to_use:.2f}')
            with self._lock:
                self._contracts[cid] = {
                    'cid':       cid,
                    'symbol':    symbol,
                    'open_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'stake':     stake_to_use,
                }

            # Subscreve para monitorar
            deriv.send({
                'proposal_open_contract': 1,
                'contract_id': cid,
                'subscribe': 1,
            })

        deriv.send({
            'proposal':      1,
            'amount':        stake_to_use,
            'basis':         'stake',
            'contract_type': 'ACCU',
            'currency':      'USD',
            'growth_rate':   GROWTH_RATE,
            'symbol':        symbol,
            'limit_order':   {'take_profit': TAKE_PROFIT},
        }, callback=on_proposal)

    def _handle_poc(self, msg):
        poc     = msg.get('proposal_open_contract', {})
        if not poc:
            return

        cid     = poc.get('contract_id')
        is_sold = poc.get('is_sold', 0)

        if not is_sold:
            return  # contrato ainda aberto

        # ── Contrato fechado ──
        with self._lock:
            info = self._contracts.pop(cid, None)

        if not info:
            return  # ja foi processado

        symbol     = info['symbol']
        sell_price = float(poc.get('sell_price', 0) or 0)
        buy_price  = float(poc.get('buy_price',  STAKE) or STAKE)
        pnl        = sell_price - buy_price

        if symbol not in self.sym_stats:
            self.sym_stats[symbol] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
        if pnl > 0:
            self.wins      += 1
            self.total_pnl += pnl
            self.sym_stats[symbol]['wins'] += 1
            self.sym_stats[symbol]['pnl']  += pnl
            result = 'WIN'
        else:
            self.losses    += 1
            self.total_pnl += pnl
            self.sym_stats[symbol]['losses'] += 1
            self.sym_stats[symbol]['pnl']    += pnl
            result = 'LOSS'

        # ── Martingale: ajusta stake do proximo trade ──
        sym_pnl_now = self.sym_stats[symbol]['pnl']
        old_stake   = self.sym_stake.get(symbol, STAKE_BASE)

        if sym_pnl_now > 0:
            # Ativo lucrativo: aplica Martingale normal
            if result == 'LOSS':
                new_stake = min(old_stake * 2, STAKE_MAX)
                log.info(f'[{symbol}] Martingale ON — LOSS — stake: ${old_stake:.2f} → ${new_stake:.2f}')
            else:
                new_stake = STAKE_BASE
                if old_stake > STAKE_BASE:
                    log.info(f'[{symbol}] Martingale ON — WIN — stake resetado: ${old_stake:.2f} → ${new_stake:.2f}')
            self.sym_stake[symbol] = new_stake
        else:
            # Ativo negativo: mantém stake base neste ativo
            self.sym_stake[symbol] = STAKE_BASE
            new_stake = STAKE_BASE

            # Se teve LOSS, redireciona o Martingale para o melhor ativo lucrativo
            if result == 'LOSS':
                best_sym = None
                best_pnl = 0.0
                for s in SYMBOLS:
                    if s == symbol:
                        continue
                    spnl = self.sym_stats.get(s, {}).get('pnl', 0.0)
                    if spnl > best_pnl:
                        best_pnl = spnl
                        best_sym = s
                if best_sym:
                    cur = self.sym_stake.get(best_sym, STAKE_BASE)
                    boosted = min(cur * 2, STAKE_MAX)
                    self.sym_stake[best_sym] = boosted
                    log.info(f'[{symbol}] LOSS negativo — Martingale redirecionado → [{best_sym}] stake: ${cur:.2f} → ${boosted:.2f}')

        log.info(
            f'[{symbol}] {result} cid={cid} PnL=${pnl:+.4f} stake_prox=${new_stake:.2f} '
            f'| {self.wins}W/{self.losses}L PnL=${self.total_pnl:+.2f}'
        )

        self.trades.appendleft({
            'symbol':      symbol,
            'cid':         cid,
            'open_time':   info.get('open_time', ''),
            'close_time':  str(poc.get('sell_time', '')),
            'stake':       round(info.get('stake', STAKE_BASE), 2),
            'pnl':         round(pnl, 4),
            'result':      result,
            'next_stake':  new_stake,
            'martingale':  sym_pnl_now > 0,
        })

        # Reabre no mesmo ativo
        threading.Timer(REOPEN_DELAY, self._open_trade, args=(symbol,)).start()

    def _watchdog(self):
        """Limpa contratos fantasmas e garante todos os ativos ativos."""
        MAX_AGE = 180  # segundos max para um ACCU com TP=$0.20
        while True:
            time.sleep(30)
            try:
                if not BOT_ACTIVE or not deriv._auth:
                    continue

                now = datetime.utcnow()

                # Limpa contratos fantasmas (abertos ha mais de MAX_AGE)
                with self._lock:
                    stale = []
                    for cid, info in self._contracts.items():
                        try:
                            opened = datetime.strptime(info['open_time'], '%Y-%m-%d %H:%M:%S UTC')
                            if (now - opened).total_seconds() > MAX_AGE:
                                stale.append((cid, info['symbol']))
                        except Exception:
                            stale.append((cid, info.get('symbol', '?')))

                    for cid, sym in stale:
                        log.warning(f'Watchdog: removendo fantasma {cid} [{sym}]')
                        self._contracts.pop(cid, None)

                # Reabre ativos que nao tem ordem aberta (e nao estao desabilitados)
                open_syms = self._symbols_open()
                opening   = self._opening.copy()
                disabled  = self.disabled_syms.copy()
                for sym in SYMBOLS:
                    if sym not in open_syms and sym not in opening and sym not in disabled:
                        log.info(f'Watchdog: [{sym}] sem ordem — reabrindo...')
                        threading.Timer(0.5, self._open_trade, args=(sym,)).start()

            except Exception as e:
                log.error(f'Watchdog erro: {e}')

trader = AccuTrader()

# ── ENDPOINTS ────────────────────────────────────────────────────────
@app.route('/')
def index():
    open_syms = trader._symbols_open()
    winrate   = (trader.wins / (trader.wins + trader.losses) * 100
                 if (trader.wins + trader.losses) > 0 else 0)
    # Stake medio ponderado atual
    stakes = [trader.sym_stake.get(s, STAKE_BASE) for s in SYMBOLS]
    avg_stake = round(sum(stakes) / len(stakes), 2) if stakes else STAKE_BASE

    return jsonify({
        'bot':          'Accumulator Multi-Asset + Martingale',
        'active':       BOT_ACTIVE,
        'ws_ready':     deriv._auth,
        'symbols':      SYMBOLS,
        'open_orders':  len(trader._contracts),
        'open_symbols': list(open_syms),
        'stake_base':   STAKE_BASE,
        'stake_max':    STAKE_MAX,
        'stake_avg':    avg_stake,
        'growth_rate':  f'{int(GROWTH_RATE*100)}%',
        'take_profit':  TAKE_PROFIT,
        'wins':         trader.wins,
        'losses':       trader.losses,
        'winrate':      round(winrate, 1),
        'total_pnl':    round(trader.total_pnl, 2),
        'last_error':   trader.last_error,
    })

@app.route('/trades')
def trades():
    return jsonify(list(trader.trades)[:100])

@app.route('/trades/<symbol>')
def trades_by_symbol(symbol):
    filtered = [t for t in trader.trades if t.get('symbol') == symbol.upper()]
    return jsonify(filtered[:50])

@app.route('/stop')
def stop_bot():
    global BOT_ACTIVE
    BOT_ACTIVE = False
    return jsonify({'ok': True, 'msg': 'Bot pausado'})

@app.route('/start')
def start_bot():
    global BOT_ACTIVE
    BOT_ACTIVE = True
    open_syms = trader._symbols_open()
    for sym in SYMBOLS:
        if sym not in open_syms:
            threading.Timer(0.3, trader._open_trade, args=(sym,)).start()
    return jsonify({'ok': True, 'msg': f'Bot iniciado em {len(SYMBOLS)} ativos'})

@app.route('/reset')
def reset():
    with trader._lock:
        cleared = list(trader._contracts.keys())
        trader._contracts.clear()
        trader._opening.clear()
    log.info(f'Reset manual: limpou {len(cleared)} contratos')
    for i, sym in enumerate(SYMBOLS):
        threading.Timer(i * 0.5, trader._open_trade, args=(sym,)).start()
    return jsonify({'ok': True, 'cleared': len(cleared), 'msg': 'Reabrindo em todos os ativos...'})

@app.route('/stats')
def stats():
    result = []
    for sym in SYMBOLS:
        s = trader.sym_stats.get(sym, {'wins': 0, 'losses': 0, 'pnl': 0.0})
        total = s['wins'] + s['losses']
        wr = round(s['wins'] / total * 100, 1) if total > 0 else 0
        active = sym not in trader.disabled_syms
        is_open = sym in trader._symbols_open()
        result.append({
            'symbol':      sym,
            'wins':        s['wins'],
            'losses':      s['losses'],
            'winrate':     wr,
            'pnl':         round(s['pnl'], 2),
            'total':       total,
            'active':      active,
            'is_open':     is_open,
            'cur_stake':   round(trader.sym_stake.get(sym, STAKE_BASE), 2),
            'martingale':  s['pnl'] > 0,
        })
    result.sort(key=lambda x: x['pnl'], reverse=True)
    return jsonify(result)

@app.route('/disable/<symbol>')
def disable_symbol(symbol):
    sym = symbol.upper()
    if sym not in SYMBOLS:
        return jsonify({'ok': False, 'msg': f'{sym} nao encontrado'}), 404
    trader.disabled_syms.add(sym)
    log.info(f'Ativo desabilitado: {sym}')
    return jsonify({'ok': True, 'msg': f'{sym} desabilitado'})

@app.route('/enable/<symbol>')
def enable_symbol(symbol):
    sym = symbol.upper()
    if sym not in SYMBOLS:
        return jsonify({'ok': False, 'msg': f'{sym} nao encontrado'}), 404
    trader.disabled_syms.discard(sym)
    log.info(f'Ativo reabilitado: {sym}')
    threading.Timer(1, trader._open_trade, args=(sym,)).start()
    return jsonify({'ok': True, 'msg': f'{sym} reabilitado'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# ── INICIALIZA (gunicorn + python direto) ────────────────────────────
if DERIV_TOKEN:
    deriv.start()
    trader.start()
else:
    log.error('DERIV_TOKEN nao configurado!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
