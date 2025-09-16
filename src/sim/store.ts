import { SimItem, SimPosition, SimPrefs } from "./types";

const QUEUE_KEY = "simQueue";
const POS_KEY   = "simPositions";
const PREF_KEY  = "simPrefs";

export function getQueue(): SimItem[] {
  try {
    const raw = localStorage.getItem(QUEUE_KEY) || localStorage.getItem("myQueue") || "[]";
    return JSON.parse(raw);
  } catch {
    return [];
  }
}
export function setQueue(list: SimItem[]) {
  localStorage.setItem(QUEUE_KEY, JSON.stringify(list));
}

export function pushToQueue(item: SimItem) {
  const list = getQueue(); list.push(item); setQueue(list);
}

export function toggleQueueDisabled(id: string, disabled: boolean) {
  const list = getQueue().map(i => i.id === id ? {...i, disabled} : i);
  setQueue(list);
}
export function removeFromQueue(id: string) {
  const list = getQueue().filter(i => i.id !== id);
  setQueue(list);
}

export function getPositions(): SimPosition[] {
  try { return JSON.parse(localStorage.getItem(POS_KEY) || "[]"); } catch { return []; }
}
export function setPositions(list: SimPosition[]) {
  localStorage.setItem(POS_KEY, JSON.stringify(list));
}

export function getPrefs(): SimPrefs {
  try { return JSON.parse(localStorage.getItem(PREF_KEY) || ""); } catch {}
  return { defaultQty: 1, autoOpenOnEnableAll: true, priceSource: "quotesBus" };
}
export function setPrefs(p: Partial<SimPrefs>) {
  const cur = getPrefs(); localStorage.setItem(PREF_KEY, JSON.stringify({...cur, ...p}));
}

/** 新开仓：用当前价作为 avgEntry */
export function openPositionFromSim(sim: SimItem, price: number, qty = getPrefs().defaultQty): SimPosition {
  const pos: SimPosition = {
    posId: (crypto as any)?.randomUUID?.() || String(Date.now()),
    fromSimId: sim.id,
    symbol: sim.symbol,
    side: sim.side as SimPosition["side"],
    strategy: sim.strategy,
    tf: sim.tf,
    qty,
    avgEntry: Number(price),
    openTime: Date.now(),
    status: "open",
  };
  const list = getPositions(); list.push(pos); setPositions(list);
  return pos;
}

export function closePosition(posId: string, price: number, alsoDisableFuture?: boolean) {
  const list = getPositions().map(p => {
    if (p.posId !== posId) return p;
    const dir = p.side === "BUY" ? 1 : -1;
    const pnl = (price - p.avgEntry) * dir * p.qty;
    return {
      ...p,
      status: "closed",
      closeTime: Date.now(),
      closePrice: Number(price),
      pnlFinal: pnl
    };
  });
  setPositions(list);

  if (alsoDisableFuture) {
    const pos = list.find(p => p.posId === posId);
    if (pos) {
      const q = getQueue().map(i => i.id === pos.fromSimId ? {...i, disabled: true} : i);
      setQueue(q);
    }
  }
}

