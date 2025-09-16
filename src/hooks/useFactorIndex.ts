export type FactorIndexPoint = {
  ts: string;
  raw: number | null;
  smoothed?: number | null;
};

export type FactorContribSeries = {
  key: 'macro' | 'policy' | 'capital' | 'geopolitics' | 'onchain' | 'sentiment' | string;
  color?: string;
  points: FactorIndexPoint[];
};

export type UseFactorIndexResult = {
  index: FactorIndexPoint[];
  contrib: FactorContribSeries[];
  asOf: string;
  isLoading: boolean;
  isError: boolean;
  source: 'api' | 'derived';
};

type HistoryItem = { ts: string; dimensions: Record<string, number | null> };

export async function useFactorIndex(params: { asset: string; granularity: string; days?: number; alpha?: number }): Promise<UseFactorIndexResult> {
  const asset = String(params.asset || 'BTC').toUpperCase();
  const granularity = String(params.granularity || 'daily').toLowerCase();
  const days = Number.isFinite(params.days as any) ? (params.days as number) : 60;
  const alpha = typeof params.alpha === 'number' ? params.alpha as number : 0.3;

  let isLoading = true;
  let isError = false;
  let source: 'api' | 'derived' = 'api';
  let asOf = '';
  let index: FactorIndexPoint[] = [];
  let contrib: FactorContribSeries[] = [];

  // 1) Try direct API
  try {
    const qs = new URLSearchParams();
    qs.set('asset', asset);
    qs.set('granularity', granularity);
    qs.set('days', String(days));
    qs.set('alpha', String(alpha));
    const resp = await fetch(`/api/factors/index?${qs.toString()}`);
    const json = await resp.json();
    if (json && json.success && json.data) {
      const d = json.data;
      index = Array.isArray(d.index) ? d.index : [];
      contrib = Array.isArray(d.contrib) ? d.contrib : [];
      asOf = d.as_of || '';
      isLoading = false;
      return { index, contrib, asOf, isLoading, isError, source: 'api' };
    }
    throw new Error('api failed');
  } catch (_) {
    // 2) Fallback: derive from /api/factors/history
    try {
      source = 'derived';
      const qs2 = new URLSearchParams();
      qs2.set('asset', asset);
      qs2.set('granularity', granularity);
      qs2.set('days', String(days));
      const resp2 = await fetch(`/api/factors/history?${qs2.toString()}`);
      const json2 = await resp2.json();
      if (!(json2 && json2.success && json2.data)) throw new Error('history failed');
      const series: HistoryItem[] = json2.data.series || [];
      asOf = json2.data.as_of || '';

      const keys: Array<FactorContribSeries['key']> = ['macro','policy','capital','geopolitics','onchain','sentiment'];
      const contribMap: Record<string, FactorIndexPoint[]> = Object.fromEntries(keys.map(k => [k, []]));
      index = [];

      for (const it of series) {
        const dims = it.dimensions || {};
        const avail = keys.filter(k => typeof dims[k] === 'number');
        if (!avail.length) {
          index.push({ ts: it.ts, raw: null, smoothed: null });
          for (const k of keys) contribMap[k].push({ ts: it.ts, raw: null, smoothed: null });
          continue;
        }
        const w = 1 / avail.length;
        let sumE = 0;
        const ctemp: Record<string, number> = {};
        for (const k of avail) {
          const s = Number(dims[k]);
          const e = (s - 50) / 50;
          const c = 50 * w * e;
          ctemp[k] = c;
          sumE += w * e;
        }
        const b = Math.max(0, Math.min(100, 50 + 50 * sumE));
        index.push({ ts: it.ts, raw: b });
        for (const k of keys) contribMap[k].push({ ts: it.ts, raw: ctemp[k] ?? null });
      }

      // EWMA smoothing
      const ewma = (arr: (number | null)[], a: number) => {
        const out: (number | null)[] = [];
        let prev: number | null = null;
        for (const v of arr) {
          if (typeof v !== 'number') { out.push(prev); continue; }
          prev = prev == null ? v : (a * v + (1 - a) * prev);
          out.push(prev);
        }
        return out;
      };
      const idxSm = ewma(index.map(p => p.raw == null ? null : Number(p.raw)), alpha);
      index = index.map((p, i) => ({ ...p, smoothed: idxSm[i] == null ? null : Number(idxSm[i]) }));

      contrib = keys.map(k => {
        const pts = contribMap[k];
        const sm = ewma(pts.map(p => p.raw == null ? null : Number(p.raw)), alpha);
        return { key: k, points: pts.map((p, i) => ({ ...p, smoothed: sm[i] == null ? null : Number(sm[i]) })) };
      });

      isLoading = false;
      return { index, contrib, asOf, isLoading, isError, source };
    } catch (e) {
      isLoading = false;
      isError = true;
      return { index: [], contrib: [], asOf: '', isLoading, isError, source };
    }
  }
}


