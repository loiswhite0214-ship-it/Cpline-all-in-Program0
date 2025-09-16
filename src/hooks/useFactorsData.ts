export type SubFactor = {
  key: string;
  score: number | null;
  weight: number;
  signal?: string | null;
  notes?: string | null;
};

export type Dimension = {
  name: 'macro' | 'policy' | 'capital' | 'geopolitics' | 'onchain' | 'sentiment' | string;
  score: number | null;
  wow: number;
  as_of: string;
  sub_factors: SubFactor[];
};

export type FactorsResponse = {
  asset: string;
  granularity: string;
  as_of: string;
  dimensions: Dimension[];
};

export type UseFactorsResult = {
  data: Dimension[];
  asOf: string;
  isLoading: boolean;
  isError: boolean;
  source: 'api' | 'mock';
};

export async function useFactorsData(params: { asset: string; granularity: string; date?: string }): Promise<UseFactorsResult> {
  const { asset, granularity, date } = params;
  let isLoading = true;
  let isError = false;
  let source: 'api' | 'mock' = 'api';
  let data: Dimension[] = [];
  let asOf = '';

  // Try API first
  try {
    const qs = new URLSearchParams();
    qs.set('asset', String(asset || 'BTC').toUpperCase());
    qs.set('granularity', String(granularity || 'daily').toLowerCase());
    if (date) qs.set('date', date);
    const resp = await fetch(`/api/factors?${qs.toString()}`);
    const json = await resp.json();
    if (json && json.success && json.data) {
      const payload = json.data as FactorsResponse;
      data = normalizeDimensions(payload.dimensions);
      asOf = payload.as_of || '';
      isLoading = false;
      console.debug('[factors]', { asset, granularity, date, source: 'api', asOf });
      return { data, asOf, isLoading, isError, source: 'api' };
    }
    throw new Error('api failed');
  } catch (_) {
    // Fallback to mock
    try {
      source = 'mock';
      const d = date || 'latest';
      const url = `/data/${String(asset || 'BTC').toUpperCase()}/${String(granularity || 'daily').toLowerCase()}/${d}.json`;
      const resp2 = await fetch(url);
      const json2 = await resp2.json();
      const payload = (json2.data ? json2.data : json2) as FactorsResponse;
      data = normalizeDimensions(payload.dimensions || []);
      asOf = payload.as_of || '';
      isLoading = false;
      console.debug('[factors]', { asset, granularity, date, source: 'mock', asOf });
      return { data, asOf, isLoading, isError, source };
    } catch (e) {
      isLoading = false;
      isError = true;
      console.debug('[factors]', { asset, granularity, date, source: 'mock', error: String(e) });
      return { data: [], asOf: '', isLoading, isError, source };
    }
  }
}

function normalizeDimensions(dims: Dimension[]): Dimension[] {
  const order: Array<Dimension['name']> = ['macro', 'policy', 'capital', 'geopolitics', 'onchain', 'sentiment'];
  const map: Record<string, Dimension> = {};
  (dims || []).forEach((d) => {
    const key = String(d.name || '').toLowerCase();
    map[key] = {
      name: key as any,
      score: d.score ?? null,
      wow: Number.isFinite(d.wow as any) ? (d.wow as number) : 0,
      as_of: d.as_of || '',
      sub_factors: Array.isArray(d.sub_factors) ? d.sub_factors : [],
    };
  });
  return order.map((k) => map[k] || { name: k, score: null, wow: 0, as_of: '', sub_factors: [] });
}


