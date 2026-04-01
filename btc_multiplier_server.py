#!/usr/bin/env python3
"""
BTC/USD Multipliers Bot — Deriv
Estratégia: Trend Following com SL/TP automático
Ativo: cryBTCUSD | Tipo: MULTUP/MULTDOWN | Multiplier: x100
"""

import os, json, time, math, threading, logging
from typing import Optional
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import websocket
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────
DERIV_TOKEN  = os.environ.get('DERIV_TOKEN', '')
DERIV_APP_ID = os.environ.get('DERIV_APP_ID', '1089')
DERIV_WS     = f'wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}'
PORT         = int(os.environ.get('PORT', 8000))
MODEL_PATH   = '/data/btc_model.pkl'

SYMBOL         = 'cryBTCUSD'
MULTIPLIER     = 100
STAKE          = 5.0    # USD (necessário para SL=4.50 funcionar)
STOP_LOSS      = 2.90   # perde no máx $2.90 por trade
TAKE_PROFIT    = 0.50   # ganha $0.50 por trade (lucro rápido)
MIN_CONF       = 0.50   # aceita qualquer sinal (invertido = ~93% acerto)
TRADE_INTERVAL = 300    # 5 minutos entre trades
INVERT_SIGNAL  = True   # INVERTE sinal: modelo erra 93% → invertido acerta 93%
TRADE_INTERVAL = 300    # segundos entre avaliações (5 min)

# ── CANDLE STORE ─────────────────────────────────────────────────
class CandleStore:
    def __init__(self):
        self.candles = {
            300:  deque(maxlen=5000),   # M5
            3600: deque(maxlen=1000),   # H1
        }
        self.lock = threading.Lock()

    def add(self, gran: int, c: dict):
        with self.lock:
            epochs = {x['epoch'] for x in self.candles[gran]}
            if c['epoch'] not in epochs:
                self.candles[gran].append(c)

    def add_batch(self, gran: int, candles: list) -> int:
        added = 0
        with self.lock:
            epochs = {x['epoch'] for x in self.candles[gran]}
            for c in candles:
                if c['epoch'] not in epochs:
                    self.candles[gran].append(c)
                    epochs.add(c['epoch'])
                    added += 1
        return added

    def get(self, gran: int, n: int = None) -> list:
        with self.lock:
            data = sorted(self.candles[gran], key=lambda x: x['epoch'])
            return data[-n:] if n else data

    def count(self) -> dict:
        with self.lock:
            return {g: len(list(q)) for g, q in self.candles.items()}

store = CandleStore()

# ── INDICADORES TÉCNICOS ─────────────────────────────────────────
def _closes(c): return [float(x['close']) for x in c]
def _highs(c):  return [float(x['high'])  for x in c]
def _lows(c):   return [float(x['low'])   for x in c]

def ema(prices, span):
    if not prices: return 0.0
    return float(pd.Series(prices).ewm(span=span, adjust=False).mean().iloc[-1])

def rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    s = pd.Series(prices)
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(alpha=1/period).mean()
    l = (-d.where(d < 0, 0.0)).ewm(alpha=1/period).mean()
    rs = g / l.replace(0, 1e-9)
    return float((100 - 100/(1+rs)).iloc[-1])

def macd_line_hist(prices):
    if len(prices) < 27: return 0.0, 0.0
    s  = pd.Series(prices)
    m  = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sg = m.ewm(span=9).mean()
    return float(m.iloc[-1]), float((m - sg).iloc[-1])

def bollinger(prices, period=20):
    if len(prices) < period: return 0.5, 0.01
    s   = pd.Series(prices)
    mn  = s.rolling(period).mean().iloc[-1]
    std = s.rolling(period).std().iloc[-1]
    if std == 0 or mn == 0: return 0.5, 0.01
    pos   = float(np.clip((prices[-1] - mn) / (2*std) + 0.5, 0, 1))
    width = float(4 * std / mn)
    return pos, width

def atr_pct(candles, period=14):
    if len(candles) < period + 1: return 0.02
    h = pd.Series(_highs(candles))
    l = pd.Series(_lows(candles))
    c = pd.Series(_closes(candles))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    price = c.iloc[-1]
    return val / price if price > 0 else 0.02

def extract_features(m5_candles: list, h1_candles: list):
    """20 features: M5 (momentum) + H1 (tendência) + tempo."""
    if len(m5_candles) < 60 or len(h1_candles) < 20:
        return None

    m5c = _closes(m5_candles)
    h1c = _closes(h1_candles)
    price = m5c[-1]
    if price == 0: return None

    # ── M5 ───────────────────────────────────────────────────────
    r14  = rsi(m5c, 14) / 100
    r7   = rsi(m5c, 7)  / 100
    ml, mh = macd_line_hist(m5c)
    ml   = math.tanh(ml  / (price * 0.001 + 1e-9))
    mh   = math.tanh(mh  / (price * 0.001 + 1e-9))
    bp5, bw5 = bollinger(m5c)
    e9   = ema(m5c, 9)
    e21  = ema(m5c, 21)
    e50  = ema(m5c, 50)
    ed1  = (e9  - e21) / (price * 0.001 + 1e-9)   # curto vs médio
    ed2  = (e21 - e50) / (price * 0.001 + 1e-9)   # médio vs longo
    pr21 = (price - e21) / (price * 0.01 + 1e-9)  # desvio do preço
    vol  = atr_pct(m5_candles[-20:])

    ret1  = (m5c[-1]/m5c[-2]  - 1) * 100 if len(m5c) >= 2  else 0
    ret6  = (m5c[-1]/m5c[-7]  - 1) * 100 if len(m5c) >= 7  else 0
    ret12 = (m5c[-1]/m5c[-13] - 1) * 100 if len(m5c) >= 13 else 0

    # ── H1 ───────────────────────────────────────────────────────
    r14h = rsi(h1c, 14) / 100
    e20h = ema(h1c, 20)
    e50h = ema(h1c, 50)
    trend_h1 = (e20h - e50h) / (h1c[-1] * 0.01 + 1e-9)
    ret6h = (h1c[-1]/h1c[-7] - 1) * 100 if len(h1c) >= 7 else 0

    # ── Tempo ────────────────────────────────────────────────────
    hr  = datetime.utcnow().hour
    sh  = math.sin(hr * math.pi / 12)
    ch  = math.cos(hr * math.pi / 12)
    dow = datetime.utcnow().weekday() / 6

    return np.array([
        r14, r7, ml, mh, bp5, bw5,
        ed1, ed2, pr21, vol,
        ret1, ret6, ret12,
        r14h, trend_h1, ret6h,
        sh, ch, dow,
        float(min(len(m5_candles), 2000) / 2000),
    ], dtype=float)

# ── LABELER: SIMULA SL/TP ────────────────────────────────────────
def label_tp_sl(candles: list, idx: int, direction: int,
                tp_pct: float = 1.5, sl_pct: float = 0.5,
                max_bars: int = 48) -> Optional[int]:
    """
    Simula resultado de um trade com SL/TP percentual.
    direction: 1=MULTUP (aposta de alta), 0=MULTDOWN (queda)
    Retorna: 1=TP atingido primeiro, 0=SL atingido primeiro, None=inconclusivo
    """
    entry = float(candles[idx]['close'])
    if direction == 1:
        tp = entry * (1 + tp_pct/100)
        sl = entry * (1 - sl_pct/100)
    else:
        tp = entry * (1 - tp_pct/100)
        sl = entry * (1 + sl_pct/100)

    for i in range(idx+1, min(idx+max_bars+1, len(candles))):
        h = float(candles[i]['high'])
        l = float(candles[i]['low'])
        if direction == 1:
            if h >= tp: return 1
            if l <= sl: return 0
        else:
            if l <= tp: return 1
            if h >= sl: return 0
    return None

# ── MODELO ML ─────────────────────────────────────────────────────
class BTCModel:
    def __init__(self):
        self.clf      = GradientBoostingClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=20, random_state=42
        )
        self.scaler   = StandardScaler()
        self.trained  = False
        self.accuracy = 0.0
        self.samples  = 0
        self.lock     = threading.Lock()
        self.live_trades = 0
        self.live_wins   = 0
        self.live_winrate = 0.0

    def train(self, m5: list, h1: list) -> bool:
        log.info(f"🧠 Treinando: {len(m5)} candles M5, {len(h1)} candles H1...")
        X, y = [], []
        skipped = 0

        for i in range(60, len(m5) - 48):
            # slice de H1 até o momento do candle M5
            epoch_i  = m5[i]['epoch']
            h1_slice = [c for c in h1 if c['epoch'] <= epoch_i][-50:]
            if len(h1_slice) < 20:
                continue

            feat = extract_features(m5[max(0, i-199):i+1], h1_slice)
            if feat is None:
                continue

            lu = label_tp_sl(m5, i, 1)   # se for MULTUP
            ld = label_tp_sl(m5, i, 0)   # se for MULTDOWN

            if lu == 1 and ld == 0:
                X.append(feat); y.append(1)   # subir claramente
            elif ld == 1 and lu == 0:
                X.append(feat); y.append(0)   # cair claramente
            else:
                skipped += 1  # ambíguo — ignora

        if len(X) < 200:
            log.warning(f"⚠️ Amostras insuficientes: {len(X)} (mínimo 200)")
            return False

        X = np.array(X)
        y = np.array(y)
        up_pct = int(y.sum() / len(y) * 100)
        log.info(f"📊 Dataset: {len(X)} amostras | ↑{up_pct}% ↓{100-up_pct}% | ignorados={skipped}")

        with self.lock:
            Xs = self.scaler.fit_transform(X)
            self.clf.fit(Xs, y)
            preds = self.clf.predict(Xs)
            self.accuracy = float(accuracy_score(y, preds))
            self.samples  = len(X)
            self.trained  = True

        log.info(f"✅ Modelo pronto! Acurácia treino={self.accuracy*100:.1f}% | {len(X)} samples")
        return True

    def predict(self, m5: list, h1: list) -> Optional[dict]:
        if not self.trained: return None
        feat = extract_features(m5[-200:], h1[-50:])
        if feat is None: return None
        with self.lock:
            try:
                Xs    = self.scaler.transform(feat.reshape(1, -1))
                pred  = int(self.clf.predict(Xs)[0])
                proba = self.clf.predict_proba(Xs)[0]
                conf  = float(max(proba))
            except Exception as e:
                log.error(f"Predict erro: {e}")
                return None
        return {
            'direction':  'MULTUP' if pred == 1 else 'MULTDOWN',
            'confidence': round(conf, 4),
            'prob_up':    round(float(proba[1] if len(proba) > 1 else proba[0]), 4),
            'prob_down':  round(float(proba[0]), 4),
        }

    def record_result(self, win: bool):
        self.live_trades += 1
        if win: self.live_wins += 1
        self.live_winrate = self.live_wins / max(self.live_trades, 1)

btc_model = BTCModel()

# ── DERIV WS PRINCIPAL ───────────────────────────────────────────
class DerivClient:
    def __init__(self):
        self._ws          = None
        self._auth        = False
        self._req         = 0
        self._lock        = threading.Lock()
        self._running     = False
        self._reconnecting = False
        self._callbacks   = {}   # req_id → fn(msg)
        self._subs        = {}   # sub_id → fn(msg)

    def _next(self) -> int:
        with self._lock:
            self._req += 1
            return self._req

    def send(self, payload: dict, callback=None) -> int:
        rid = self._next()
        payload['req_id'] = rid
        if callback:
            self._callbacks[rid] = callback
        if self._ws and self._ws.sock and self._ws.sock.connected:
            try:
                self._ws.send(json.dumps(payload))
            except Exception as e:
                log.warning(f"WS send erro: {e}")
                self._callbacks.pop(rid, None)
        return rid

    def _on_open(self, ws):
        self._reconnecting = False
        log.info("🔌 Conectado à Deriv — autorizando...")
        self.send({'authorize': DERIV_TOKEN})

    def _on_message(self, ws, raw):
        try:
            msg   = json.loads(raw)
            mtype = msg.get('msg_type', '')
            rid   = msg.get('req_id')
            err   = msg.get('error')

            if err:
                log.warning(f"⚠️ Deriv: {err.get('message','?')}")
                if rid: self._callbacks.pop(rid, None)
                return

            if mtype == 'authorize':
                self._auth = True
                log.info("✅ Autorizado")
                return

            if rid and rid in self._callbacks:
                cb = self._callbacks.pop(rid)
                cb(msg)

            sub_id = msg.get('subscription', {}).get('id')
            if sub_id and sub_id in self._subs:
                self._subs[sub_id](msg)

        except Exception as e:
            log.error(f"WS handler erro: {e}")

    def _on_error(self, ws, err):
        log.error(f"WS erro: {err}")
        self._auth = False

    def _on_close(self, ws, code, msg):
        self._auth = False
        if self._reconnecting:
            return  # já tem outra reconexão em andamento
        self._reconnecting = True
        log.warning(f"WS fechado ({code}). Reconectando em 10s...")

        def do_reconnect():
            time.sleep(10)
            if self._running:
                try:
                    self._connect()
                except Exception as e:
                    log.error(f"Reconexão erro: {e}")
                    self._reconnecting = False
        threading.Thread(target=do_reconnect, daemon=True).start()

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            DERIV_WS,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
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
    def is_ready(self) -> bool:
        return self._auth

deriv = DerivClient()

# ── DOWNLOAD DE 1 ANO ────────────────────────────────────────────
class Downloader:
    BATCH = 5000

    def __init__(self):
        self._status   = 'idle'
        self._progress = {}
        self._lock     = threading.Lock()

    def _fetch_batch(self, gran: int, end_epoch: int) -> Optional[list]:
        """Busca um batch de candles até end_epoch."""
        event  = threading.Event()
        result = {}

        def cb(msg):
            result['data'] = msg.get('candles', [])
            event.set()

        deriv.send({
            'ticks_history':     SYMBOL,
            'style':             'candles',
            'granularity':       gran,
            'count':             self.BATCH,
            'end':               end_epoch,
            'adjust_start_time': 1,
        }, callback=cb)

        if not event.wait(timeout=30):
            return None
        return result.get('data')

    def download(self, gran: int, months: int = 12) -> list:
        label     = f"M{gran//60}" if gran < 3600 else f"H{gran//3600}"
        end       = int(time.time())
        start     = end - months * 30 * 24 * 3600
        all_c     = []
        batch_num = 0

        log.info(f"📥 [{label}] Iniciando download de {months} meses...")

        while end > start:
            batch_num += 1
            batch = self._fetch_batch(gran, end)

            if not batch:
                log.warning(f"[{label}] Batch {batch_num} sem dados — encerrando")
                break

            # filtra dentro do range
            batch = [c for c in batch if c['epoch'] >= start]
            if not batch:
                break

            all_c.extend(batch)
            oldest = min(c['epoch'] for c in batch)
            end    = oldest - 1

            with self._lock:
                self._progress[label] = {
                    'batches': batch_num,
                    'candles': len(all_c),
                    'from':    datetime.utcfromtimestamp(oldest).strftime('%Y-%m-%d'),
                }

            log.info(f"  [{label}] batch {batch_num}: +{len(batch)} | total={len(all_c)} | desde={datetime.utcfromtimestamp(oldest).strftime('%Y-%m-%d')}")
            time.sleep(0.5)   # respeita rate limit

        # ordena e remove duplicatas
        all_c.sort(key=lambda x: x['epoch'])
        seen, uniq = set(), []
        for c in all_c:
            if c['epoch'] not in seen:
                seen.add(c['epoch']); uniq.append(c)

        log.info(f"✅ [{label}] {len(uniq)} candles únicos")
        return uniq

    def run(self):
        with self._lock:
            self._status = 'downloading'

        try:
            # aguarda autorização
            for _ in range(60):
                if deriv.is_ready: break
                time.sleep(1)
            if not deriv.is_ready:
                raise RuntimeError("Sem conexão com Deriv")

            h1 = self.download(3600, months=12)
            store.add_batch(3600, h1)

            m5 = self.download(300, months=12)
            store.add_batch(300, m5)

            if len(m5) < 500:
                raise RuntimeError(f"Dados insuficientes: {len(m5)} candles M5")

            with self._lock:
                self._status = 'training'

            ok = btc_model.train(m5, h1)
            if ok:
                save_model()
                with self._lock:
                    self._status = 'done'
                log.info("🎉 Download e treinamento concluídos!")
            else:
                with self._lock:
                    self._status = 'error'

        except Exception as e:
            log.error(f"Downloader erro: {e}")
            with self._lock:
                self._status = 'error'

    def start(self):
        if self._status == 'running': return
        threading.Thread(target=self.run, daemon=True).start()

    @property
    def status(self):
        with self._lock: return self._status

    @property
    def progress(self):
        with self._lock: return dict(self._progress)

downloader = Downloader()

# ── LIVE FEED (candles ao vivo) ──────────────────────────────────
class LiveFeed:
    def subscribe(self):
        for gran in [300, 3600]:
            self._sub(gran)

    def _sub(self, gran: int):
        label = f"M{gran//60}" if gran < 3600 else f"H{gran//3600}"

        def on_update(msg):
            if msg.get('msg_type') == 'ohlc':
                o = msg['ohlc']
                store.add(gran, {
                    'epoch': int(o['open_time']),
                    'open':  float(o['open']),
                    'high':  float(o['high']),
                    'low':   float(o['low']),
                    'close': float(o['close']),
                })

        def on_initial(msg):
            for c in msg.get('candles', []):
                store.add(gran, c)
            sub_id = msg.get('subscription', {}).get('id')
            if sub_id:
                deriv._subs[sub_id] = on_update
            log.info(f"📡 Live feed {label} ativo")

        # M5: 500 candles = ~41h | H1: 200 candles = ~8 dias
        initial_count = 500 if gran == 300 else 200
        deriv.send({
            'ticks_history': SYMBOL,
            'style':         'candles',
            'granularity':   gran,
            'count':         initial_count,
            'end':           'latest',
            'subscribe':     1,
        }, callback=on_initial)

live_feed = LiveFeed()

# ── TRADER (Multipliers) ──────────────────────────────────────────
class MultiplierTrader:
    def __init__(self):
        self._ws      = None
        self._auth    = False
        self._running = False
        self._lock    = threading.Lock()
        self._req     = 0
        self._pending = {}          # req_id/contract_id → info
        self._open    = None        # contract_id em aberto
        self.trades   = deque(maxlen=200)
        self.wins     = 0
        self.losses   = 0
        self.last_error = None        # último erro da Deriv

    def _next(self) -> int:
        with self._lock:
            self._req += 1
            return self._req

    def _send(self, payload: dict):
        if self._ws and self._ws.sock and self._ws.sock.connected:
            try:
                payload['req_id'] = self._next()
                self._ws.send(json.dumps(payload))
            except Exception as e:
                log.warning(f"Trader send erro: {e}")

    def _on_open(self, ws):
        log.info("💰 Trader WS aberto — enviando authorize...")
        try:
            payload = json.dumps({'authorize': DERIV_TOKEN, 'req_id': self._next()})
            ws.send(payload)
            log.info("💰 Authorize enviado ao Trader WS")
        except Exception as e:
            log.error(f"💰 ERRO ao enviar authorize no Trader: {e}")

    def _on_message(self, ws, raw):
        try:
            msg   = json.loads(raw)
            mtype = msg.get('msg_type', '')
            err   = msg.get('error')
            rid   = msg.get('req_id')

            if err:
                err_msg = f"{err.get('code','?')} — {err.get('message','?')} | msg_type={mtype}"
                log.warning(f"⚠️ Trader ERRO: {err_msg}")
                self.last_error = {'code': err.get('code'), 'message': err.get('message'), 'type': mtype, 'time': datetime.utcnow().isoformat()}
                self._pending.pop('pending_proposal', None)
                self._pending.pop('pending_buy', None)
                return

            if mtype == 'authorize':
                self._auth = True
                log.info("✅ Trader autorizado")
                # re-subscreve contrato aberto se WS reconectou com contrato pendente
                with self._lock:
                    open_cid = self._open
                if open_cid:
                    log.info(f"🔄 Re-subscrevendo contrato {open_cid} após reconexão...")
                    self._send({'proposal_open_contract': 1, 'contract_id': open_cid, 'subscribe': 1})
                return

            if mtype == 'proposal':
                prop    = msg.get('proposal', {})
                prop_id = prop.get('id')
                info    = self._pending.pop('pending_proposal', None)
                if prop_id and info:
                    log.info(f"📋 Proposta: id={prop_id} ask={prop.get('ask_price')} comm={prop.get('commission')}")
                    self._pending['pending_buy'] = info
                    self._send({'buy': prop_id, 'price': float(prop.get('ask_price', STAKE))})
                return

            if mtype == 'buy':
                buy  = msg.get('buy', {})
                cid  = buy.get('contract_id')
                info = self._pending.pop('pending_buy', None)
                if cid and info:
                    with self._lock:
                        self._pending[cid] = info
                        self._open = cid
                    log.info(f"✅ Contrato aberto {cid} | {info['direction']} | stake=${STAKE} SL=${STOP_LOSS} TP=${TAKE_PROFIT}")
                    self._send({'proposal_open_contract': 1, 'contract_id': cid, 'subscribe': 1})
                return

            if mtype == 'proposal_open_contract':
                poc = msg.get('proposal_open_contract', {})
                cid = poc.get('contract_id')
                if not poc.get('is_expired') and not poc.get('is_sold'):
                    return   # ainda aberto

                with self._lock:
                    info = self._pending.pop(cid, None)
                    self._open = None

                if not info: return

                profit = float(poc.get('profit', 0))
                win    = profit > 0

                if win: self.wins   += 1
                else:   self.losses += 1

                btc_model.record_result(win)

                self.trades.append({
                    'contract_id': cid,
                    'direction':   info['direction'],
                    'confidence':  info.get('confidence', 0),
                    'profit':      round(profit, 2),
                    'result':      'win' if win else 'loss',
                    'time':        datetime.utcnow().strftime('%H:%M:%S UTC'),
                })

                total = self.wins + self.losses
                log.info(
                    f"{'✅ WIN' if win else '❌ LOSS'} {info['direction']} "
                    f"profit={profit:+.2f} | "
                    f"total={total} W={self.wins} L={self.losses} "
                    f"WR={self.wins/max(total,1)*100:.1f}%"
                )

        except Exception as e:
            log.error(f"Trader handler erro: {e}")

    def _on_error(self, ws, err):
        log.error(f"Trader WS erro: {err}")

    def _on_close(self, ws, code, msg):
        self._auth = False
        log.warning(f"Trader WS fechado ({code}). Reconectando em 15s...")
        def _reconnect():
            time.sleep(15)
            if self._running:
                self._connect()
        threading.Thread(target=_reconnect, daemon=True).start()

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            DERIV_WS,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        threading.Thread(
            target=self._ws.run_forever,
            kwargs={'ping_interval': 30, 'ping_timeout': 10},
            daemon=True
        ).start()

    def open_trade(self, direction: str, confidence: float):
        if not self._auth:
            log.warning("Trader não autorizado — aguardando reconexão")
            return
        if self._open:
            log.info(f"⏭ Contrato {self._open} ainda aberto — aguardando fechar")
            return
        if 'pending_proposal' in self._pending or 'pending_buy' in self._pending:
            log.info("⏭ Já tem proposta/compra pendente — aguardando")
            return

        # ── INVERSÃO DE SINAL ─────────────────────────────────
        original = direction
        if INVERT_SIGNAL:
            direction = 'MULTDOWN' if direction == 'MULTUP' else 'MULTUP'
            log.info(f"🔄 Sinal invertido: {original} → {direction}")

        self._pending['pending_proposal'] = {
            'direction':  direction,
            'confidence': confidence,
        }
        self._send({
            'proposal':      1,
            'contract_type': direction,
            'symbol':        SYMBOL,
            'amount':        STAKE,
            'basis':         'stake',
            'multiplier':    MULTIPLIER,
            'currency':      'USD',
            'limit_order': {
                'stop_loss':   STOP_LOSS,
                'take_profit': TAKE_PROFIT,
            },
        })
        log.info(f"🎯 Abrindo {direction} (modelo={original}) conf={confidence:.1%}")

    def _trade_loop(self):
        log.info("⏳ Aguardando modelo e dados para iniciar trading...")
        while self._running:
            if btc_model.trained and len(store.get(300)) >= 100:
                break
            time.sleep(10)

        log.info("🚀 Iniciando loop de trading BTC/USD Multipliers")

        while self._running:
            try:
                if not self._auth:
                    log.warning("⏳ Trade loop: Trader NÃO autorizado — aguardando auth...")
                    time.sleep(10)
                    continue

                m5 = store.get(300)
                h1 = store.get(3600)

                if len(m5) < 100 or len(h1) < 20:
                    log.info(f"⏳ Dados: M5={len(m5)} H1={len(h1)} — aguardando mais candles")
                    time.sleep(30)
                    continue

                signal = btc_model.predict(m5, h1)

                if not signal:
                    time.sleep(TRADE_INTERVAL)
                    continue

                conf = signal['confidence']
                dir_ = signal['direction']

                log.info(
                    f"📡 Sinal: {dir_} | conf={conf:.1%} | "
                    f"↑{signal['prob_up']:.1%} ↓{signal['prob_down']:.1%} | "
                    f"M5={len(m5)} H1={len(h1)}"
                )

                if conf >= MIN_CONF:
                    self.open_trade(dir_, conf)
                else:
                    log.info(f"⏭ conf {conf:.1%} < {MIN_CONF:.1%} — sinal fraco, aguardando")

                time.sleep(TRADE_INTERVAL)

            except Exception as e:
                log.error(f"Trade loop erro: {e}")
                time.sleep(30)

    def start(self):
        if not DERIV_TOKEN:
            log.warning("⚠️ DERIV_TOKEN não configurado — use /config para definir")
            return
        self._running = True
        self._connect()
        threading.Thread(target=self._trade_loop, daemon=True).start()
        log.info("💰 Trader iniciado")

    @property
    def is_running(self) -> bool:
        return self._running and self._auth

    @property
    def winrate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total * 100 if total > 0 else 0.0

trader = MultiplierTrader()

# ── PERSISTÊNCIA ─────────────────────────────────────────────────
def save_model():
    try:
        if not btc_model.trained: return
        data = {
            'clf':         btc_model.clf,
            'scaler':      btc_model.scaler,
            'accuracy':    btc_model.accuracy,
            'samples':     btc_model.samples,
            'live_trades': btc_model.live_trades,
            'live_wins':   btc_model.live_wins,
        }
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(data, MODEL_PATH)
        log.info(f"💾 Modelo salvo em {MODEL_PATH}")
    except Exception as e:
        log.error(f"Save erro: {e}")

def load_model():
    try:
        if not os.path.exists(MODEL_PATH): return
        data = joblib.load(MODEL_PATH)
        btc_model.clf         = data['clf']
        btc_model.scaler      = data['scaler']
        btc_model.accuracy    = data.get('accuracy', 0.0)
        btc_model.samples     = data.get('samples', 0)
        btc_model.trained     = True
        btc_model.live_trades = data.get('live_trades', 0)
        btc_model.live_wins   = data.get('live_wins', 0)
        btc_model.live_winrate = btc_model.live_wins / max(btc_model.live_trades, 1)
        log.info(f"✅ Modelo carregado: {btc_model.samples} samples | acc={btc_model.accuracy*100:.1f}%")
    except Exception as e:
        log.error(f"Load erro: {e}")

def auto_save():
    while True:
        time.sleep(300)
        save_model()

# ── FLASK API ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/reset-contract', methods=['POST'])
def reset_contract():
    """Desbloqueia contrato travado manualmente."""
    old = trader._open
    with trader._lock:
        trader._open = None
        trader._pending.pop(old, None) if old else None
    log.info(f"🔧 Contrato {old} removido manualmente via /reset-contract")
    return jsonify({'ok': True, 'cleared': old})

@app.route('/reset-stats', methods=['POST'])
def reset_stats():
    """Zera contadores de performance (novo começo com sinal invertido)."""
    trader.wins = 0
    trader.losses = 0
    trader.trades.clear()
    btc_model.live_trades = 0
    btc_model.live_wins = 0
    btc_model.live_winrate = 0.0
    log.info("🔄 Contadores zerados — recomeçando do zero (sinal invertido)")
    return jsonify({'ok': True, 'message': 'Stats zeradas'})

@app.route('/stats')
def stats():
    total = trader.wins + trader.losses
    return jsonify({
        'symbol':      SYMBOL,
        'multiplier':  MULTIPLIER,
        'stake':       STAKE,
        'stop_loss':   STOP_LOSS,
        'take_profit': TAKE_PROFIT,
        'min_conf':       MIN_CONF,
        'invert_signal':  INVERT_SIGNAL,
        'connected':      deriv.is_ready,
        'trader_running': trader.is_running,
        'trader_auth':    trader._auth,
        'trader_ws_ok':   bool(trader._ws and trader._ws.sock and trader._ws.sock.connected) if trader._ws else False,
        'open_contract':  trader._open,
        'pending':        list(trader._pending.keys()),
        'download': {
            'status':   downloader.status,
            'progress': downloader.progress,
        },
        'candles': store.count(),
        'model': {
            'trained':      btc_model.trained,
            'samples':      btc_model.samples,
            'accuracy':     round(btc_model.accuracy * 100, 2),
            'live_trades':  btc_model.live_trades,
            'live_wins':    btc_model.live_wins,
            'live_winrate': round(btc_model.live_winrate * 100, 2),
        },
        'performance': {
            'total':   total,
            'wins':    trader.wins,
            'losses':  trader.losses,
            'winrate': round(trader.winrate, 2),
            'profit':  round(sum(t['profit'] for t in trader.trades), 2),
        },
        'recent_trades': list(trader.trades)[-10:],
    })

@app.route('/start-download', methods=['POST'])
def start_download():
    if downloader.status in ('running', 'downloading', 'training'):
        return jsonify({'error': 'Download já em andamento', 'status': downloader.status}), 400
    downloader.start()
    return jsonify({'status': 'started', 'message': 'Download de 1 ano de BTC/USD iniciado'})

@app.route('/config', methods=['POST'])
def config():
    global DERIV_TOKEN
    data = request.json or {}
    if 'token' in data:
        DERIV_TOKEN = data['token']
        log.info("🔑 Token configurado via /config")
    return jsonify({'ok': True})

@app.route('/reconnect', methods=['POST'])
def reconnect():
    """Força reconexão com a Deriv."""
    log.info("🔄 Reconexão forçada via /reconnect")
    try:
        if deriv._ws:
            deriv._ws.close()
    except: pass
    deriv._auth = False
    deriv._reconnecting = False
    time.sleep(2)
    deriv._connect()
    return jsonify({'status': 'reconnecting'})

@app.route('/debug')
def debug():
    """Diagnóstico: testa predição e mostra estado completo."""
    m5 = store.get(300)
    h1 = store.get(3600)
    signal = None
    predict_error = None
    try:
        if btc_model.trained and len(m5) >= 100 and len(h1) >= 20:
            signal = btc_model.predict(m5, h1)
    except Exception as e:
        predict_error = str(e)

    ws_ok = False
    try:
        ws_ok = bool(trader._ws and trader._ws.sock and trader._ws.sock.connected)
    except:
        pass

    return jsonify({
        'trader_auth':     trader._auth,
        'trader_ws_ok':    ws_ok,
        'trader_running':  trader._running,
        'trader_open':     trader._open,
        'trader_pending':  list(trader._pending.keys()),
        'deriv_ready':     deriv.is_ready,
        'model_trained':   btc_model.trained,
        'candles_m5':      len(m5),
        'candles_h1':      len(h1),
        'signal':          signal,
        'predict_error':   predict_error,
        'min_conf':        MIN_CONF,
        'would_trade':     signal is not None and signal['confidence'] >= MIN_CONF,
        'deriv_token_set': bool(DERIV_TOKEN),
        'last_error':      trader.last_error,
    })

@app.route('/force-trade', methods=['GET', 'POST'])
def force_trade():
    """Força uma trade de teste."""
    m5 = store.get(300)
    h1 = store.get(3600)
    if not btc_model.trained:
        return jsonify({'error': 'Modelo não treinado'}), 400
    if len(m5) < 100:
        return jsonify({'error': f'Poucas candles M5: {len(m5)}'}), 400
    signal = btc_model.predict(m5, h1)
    if not signal:
        return jsonify({'error': 'Predict retornou None'}), 400
    trader.open_trade(signal['direction'], signal['confidence'])
    return jsonify({'ok': True, 'signal': signal, 'inverted': INVERT_SIGNAL})

@app.route('/')
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.utcnow().isoformat()})

@app.route('/monitor')
def monitor():
    """Serve o monitor HTML diretamente pelo servidor."""
    monitor_path = os.path.join(os.path.dirname(__file__), 'btc_monitor.html')
    if os.path.exists(monitor_path):
        return send_file(monitor_path)
    return "Monitor não encontrado", 404

# ── STARTUP (roda com gunicorn E python direto) ───────────────────
def _startup():
    try:
        log.info("=" * 60)
        log.info("🚀 BTC/USD Multipliers Bot — Deriv")
        log.info(f"   Símbolo:     {SYMBOL}")
        log.info(f"   Multiplier:  x{MULTIPLIER}")
        log.info(f"   Stake:       ${STAKE}")
        log.info(f"   Stop Loss:   ${STOP_LOSS}  ({STOP_LOSS/STAKE*100:.0f}% do stake)")
        log.info(f"   Take Profit: ${TAKE_PROFIT} ({TAKE_PROFIT/STAKE*100:.0f}% do stake)")
        log.info(f"   RR:          SL=${STOP_LOSS} TP=${TAKE_PROFIT} → precisa acertar >{STOP_LOSS/(STOP_LOSS+TAKE_PROFIT)*100:.0f}% para lucrar")
        log.info(f"   Confiança:   {MIN_CONF*100:.0f}%")
        log.info("=" * 60)

        load_model()

        if DERIV_TOKEN:
            deriv.start()
            time.sleep(3)
            trader.start()
        else:
            log.warning("⚠️ DERIV_TOKEN vazio — aguardando /config")

        threading.Thread(target=auto_save, daemon=True).start()

        # Watchdog: reconecta Deriv se cair
        def watchdog():
            time.sleep(60)
            while True:
                try:
                    if not deriv.is_ready and DERIV_TOKEN and not deriv._reconnecting:
                        log.warning("🔄 Watchdog: Deriv desconectado — reconectando...")
                        deriv._reconnecting = True
                        try:
                            if deriv._ws: deriv._ws.close()
                        except: pass
                        time.sleep(5)
                        try:
                            deriv._connect()
                        except Exception as e:
                            log.error(f"Watchdog reconexão erro: {e}")
                            deriv._reconnecting = False
                except Exception as e:
                    log.error(f"Watchdog erro: {e}")
                    deriv._reconnecting = False
                time.sleep(30)
        threading.Thread(target=watchdog, daemon=True).start()
    except Exception as e:
        log.error(f"❌ STARTUP ERRO: {e}")
        import traceback
        log.error(traceback.format_exc())

    if not btc_model.trained and DERIV_TOKEN:
        log.info("📥 Sem modelo salvo — iniciando download automático em 5s...")
        time.sleep(5)
        downloader.start()

    # live feed — aguarda conexão + download e subscreve
    def await_and_subscribe():
        # espera download se estiver rodando
        while downloader.status not in ('done', 'error', 'idle'):
            time.sleep(5)
        # espera Deriv conectar (até 120s)
        for _ in range(40):
            if deriv.is_ready:
                break
            time.sleep(3)
        if deriv.is_ready:
            log.info("📡 Inscrevendo live feed M5 + H1...")
            live_feed.subscribe()
        else:
            log.warning("⚠️ Deriv não conectou — live feed não inscrito. Watchdog vai tentar reconectar.")

    threading.Thread(target=await_and_subscribe, daemon=True).start()

# Roda em background para não bloquear o gunicorn durante o import
threading.Thread(target=_startup, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
# redeploy Mon Mar 23 23:02:57     2026
