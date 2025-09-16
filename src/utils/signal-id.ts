export const signalId = (s: {
  symbol: string; strategy: string; side: 'BUY'|'SELL'; tf: string;
  entry: number; time?: string;
}) => {
  const t = s.time || '';
  return `${s.symbol}|${s.strategy}|${s.side}|${s.tf}|${t}`;
};


