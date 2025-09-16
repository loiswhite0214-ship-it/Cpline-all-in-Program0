export type Side = "BUY"|"SELL";
export type SignalDto = {
  id:string; symbol:string; side:Side; timeframe:string; strategy:string;
  queued_at?:string; ref_price?:number;
};
export type PositionDto = {
  id:string; symbol:string; side:Side; timeframe:string; strategy:string;
  qty:number; avg_price:number; mark_price?:number|null;
  opened_at:string; pnl?:number|null; pnl_pct?:number|null;
};

export function unique<T, K extends keyof any>(arr:T[], key:(x:T)=>K):T[]{
  const seen = new Set<K>(); const out:T[]=[];
  for(const it of arr){ const k=key(it); if(!seen.has(k)){ seen.add(k); out.push(it);} }
  return out;
}

export function savePref<T>(k:string, v:T){ try{ localStorage.setItem(k, JSON.stringify(v)); }catch{} }
export function loadPref<T>(k:string, d:T):T{ try{ const v=localStorage.getItem(k); return v?JSON.parse(v) as T: d;}catch{ return d; } }

