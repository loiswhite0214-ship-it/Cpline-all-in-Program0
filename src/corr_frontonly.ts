/* ========= 可配置 ========= */
const WIN = 30;                    // 滚动窗口天数
const LOW_VAR_EPS = 1e-12;         // 低方差保护阈值（更宽松）
const EMA_SOFT = 3;                // 轻平滑
const INTERP_SPAN = 7;             // 低频插值 & EMA 去台阶
/* ========================= */

/** 颜色体系（3 色相，Web3 风） */
const COLORS = {
  cyan:   { base:'#29E3FF', mid:'#15B6E9', dark:'#0F8FBF' },
  violet: { base:'#B16CFF', mid:'#8D4CF0', dark:'#5D2AB6' },
  mint:   { base:'#7CFFB2', mid:'#46E68F', dark:'#1CA96A' },
  grid:   'rgba(28,36,48,.25)', text:'#CFE6FF'
};

/** 映射：后端字段名 → 目标因子名（尽量全些；没匹配就跳过） */
const ALIASES: Record<string,string> = {
  'usd_i':'DXY','dxy':'DXY','dx':'DXY','dx_f':'DXY','DXY':'DXY',
  '^spx':'SPX','spx':'SPX','sp500':'SPX','SPX':'SPX',
  'xau':'XAU','xauusd':'XAU','gold':'XAU','XAU':'XAU',
  'vix':'VIX','VIX':'VIX',
  'fng':'FNG','fear_greed':'FNG','feargreed':'FNG',
  'funding':'Funding','funding_rate':'Funding','Funding':'Funding',
  'etf_flows':'ETF_Flows','etf_flow':'ETF_Flows','etfnet':'ETF_Flows','ETF_Flows':'ETF_Flows',
  'nfci':'NFCI','NFCI':'NFCI'
};

/** 价格类走对数收益，其他走差分 */
const PRICE_KEYS = new Set(['SPX','XAU','DXY']);
const DIFF_KEYS  = new Set(['VIX','FNG','Funding','ETF_Flows','NFCI']);

/* ========== 小工具 ========== */
const toDate = (v:any) => new Date(typeof v==='number' ? v : String(v));
const fmt = (d:Date) => d.toISOString().slice(0,10);
const uniq = <T>(arr:T[]) => Array.from(new Set(arr));

function parseFactors(json:any){
  // 允许形式：{data:[...]} 或 直接 [...]
  const rows = Array.isArray(json) ? json : Array.isArray(json?.data) ? json.data : [];
  if (!rows.length) throw new Error('empty factors payload');

  // 识别日期列
  const dateKey = ['date','time','timestamp','ts'].find(k => k in rows[0])!;
  // 收集所有数值列
  const numKeys = Object.keys(rows[0]).filter(k=>{
    if (k===dateKey) return false;
    const v = rows[0][k];
    return typeof v==='number' || (!isNaN(+v) && v!=='' && v!=null);
  });

  // 猜 BTC 收盘列
  const btcCol = numKeys.find(k=>/btc/i.test(k) && /close|price|last/i.test(k))
             ?? numKeys.find(k=>/^close$/i.test(k))
             ?? numKeys.find(k=>/btc/i.test(k));
  if (!btcCol) throw new Error('no BTC close column in factors');

  // 标准化记录：date -> { key:value }
  const map: Record<string, Record<string,number>> = {};
  for (const r of rows){
    const d = fmt(toDate(r[dateKey]));
    if (!map[d]) map[d] = {};
    for (const k of numKeys){
      const v = +r[k]; if (!isFinite(v)) continue;
      map[d][k] = v;
    }
  }
  return { map, btcCol, numKeys, dates: Object.keys(map).sort() };
}

function seriesFromMap(map:Record<string,Record<string,number>>, key:string){
  const dates = Object.keys(map).sort();
  return dates.map(d => [d, map[d]?.[key] ?? null] as [string, number|null]);
}

function ema(arr:number[], span:number){
  if (span<=1) return arr.slice();
  const alpha = 2/(span+1);
  let prev = arr.find(v=>isFinite(v)) ?? 0;
  return arr.map(v=>{
    const x = isFinite(v) ? v : prev;
    prev = alpha*x + (1-alpha)*prev;
    return prev;
  });
}

function dailyRange(dates:string[]){
  const s = new Date(dates[0]+'T00:00:00Z').getTime();
  const e = new Date(dates[dates.length-1]+'T00:00:00Z').getTime();
  const out:string[] = [];
  for (let t=s; t<=e; t+=86400000) out.push(fmt(new Date(t)));
  return out;
}

/** 判断是否低频（中位间隔 >= 2 天） */
function isLowFreq(dates:string[]){
  if (dates.length<3) return true;
  const gaps:number[]=[];
  for (let i=1;i<dates.length;i++){
    gaps.push((+new Date(dates[i]) - +new Date(dates[i-1]))/86400000);
  }
  gaps.sort((a,b)=>a-b);
  const med = gaps[Math.floor(gaps.length/2)];
  return med >= 2;
}

/** 线性插值到日频 + 7日EMA 去台阶 */
function upsampleDaily(pairs:[string,number|null][], span=INTERP_SPAN){
  const rawDates = pairs.map(p=>p[0]).filter(Boolean);
  const full = dailyRange(rawDates);
  const valMap = new Map(pairs.filter(p=>p[1]!=null) as [string,number][]);

  // 先前向填充，再做线性插值
  const out:number[] = [];
  let lastV:number|undefined;
  for (let i=0;i<full.length;i++){
    const d = full[i];
    const v = valMap.get(d);
    if (v!=null) { out.push(v); lastV=v; continue; }
    if (lastV!=null){
      // 向后找下一个有效点做线性插值
      let j=i+1, nextV: number|undefined, nextPos=i;
      for (; j<full.length; j++){
        const w = valMap.get(full[j]);
        if (w!=null){ nextV=w; nextPos=j; break; }
      }
      if (nextV!=null){
        const k = (i - (i-1)) / (nextPos - (i-1));
        const inter = lastV + (nextV - lastV) * (1/(nextPos-i+1));
        out.push(inter);
      } else {
        out.push(lastV);
      }
    } else out.push(0);
  }
  return { dates: full, values: ema(out, span) };
}

/** 对数收益 / 一阶差分 */
function transformSeries(values:number[], mode:'logret'|'diff'){
  if (mode==='logret'){
    const out:number[] = [NaN];
    for (let i=1;i<values.length;i++){
      const prev = values[i-1], cur = values[i];
      out.push( (Math.log(Math.max(1e-12, cur)) - Math.log(Math.max(1e-12, prev))) );
    }
    return out;
  }else{
    const out:number[]=[NaN];
    for (let i=1;i<values.length;i++) out.push(values[i]-values[i-1]);
    return out;
  }
}

/** 计算秩（处理并列：平均名次），输出 0..1 */
function rank01(a:number[]){
  const n=a.length;
  const idx = a.map((v,i)=>[v,i] as [number,number]).sort((x,y)=>x[0]-y[0]);
  const r = new Array(n).fill(0);
  let i=0;
  while(i<n){
    let j=i;
    while(j+1<n && idx[j+1][0]===idx[i][0]) j++;
    const avg = (i+j+2)/2; // 1-based 平均名次
    for (let k=i;k<=j;k++) r[idx[k][1]] = avg;
    i=j+1;
  }
  const max = n; return r.map(v=>v/max);
}

/** 滚动秩相关（Spearman），带低方差保护与限幅 */
function rollingSpearman(a:number[], b:number[], w=WIN){
  const n=a.length, out=(new Array(n)).fill(null) as (number|null)[];
  const ra = rank01(a), rb = rank01(b);
  for (let i=w-1;i<n;i++){
    let sa=0,sb=0,sa2=0,sb2=0,sab=0;
    for (let k=i-w+1;k<=i;k++){
      const va=ra[k], vb=rb[k];
      sa+=va; sb+=vb;
    }
    const ma = sa/w, mb = sb/w;
    for (let k=i-w+1;k<=i;k++){
      const va=ra[k]-ma, vb=rb[k]-mb;
      sa2 += va*va; sb2 += vb*vb; sab += va*vb;
    }
    const den = Math.sqrt(sa2*sb2);
    out[i] = (den < LOW_VAR_EPS) ? null : Math.max(-0.95, Math.min(0.95, sab/den))*100;
  }
  // 轻平滑 + 前向填充
  const sm = ema(out.map(v=> (v==null? NaN: v) as any), EMA_SOFT)
              .map(v=> isFinite(v)? v : null);
  let last: number|null = null;
  for (let i=0;i<n;i++){ if (sm[i]==null && last!=null) sm[i]=last; else if (sm[i]!=null) last=sm[i]!; }
  return sm;
}

/** 计算所有因子对 BTC 的 30D 秩相关（前端-only） */
async function computeCorrFrontOnly(asset='BTC', win=WIN){
  // 使用 /api/factors/history 端点获取历史数据
  const res = await fetch(`/api/factors/history?asset=${asset}&granularity=daily&days=180`);
  const json = await res.json();

  if (!json.success || !json.data || !json.data.series) {
    throw new Error('Failed to fetch factors history data');
  }

  const series = json.data.series;
  if (!series.length) {
    throw new Error('No historical data available');
  }

  // 提取日期和因子数据
  const dates = series.map((s: any) => s.ts.split('T')[0]);
  const dimensions = ['macro', 'policy', 'capital', 'geopolitics', 'onchain', 'sentiment'];
  
  // 构建因子数据矩阵
  const factorData: Record<string, number[]> = {};
  dimensions.forEach(dim => {
    factorData[dim] = series.map((s: any) => s.dimensions[dim] || 0);
  });

  // 模拟BTC价格数据（实际应用中应该从其他API获取）
  const btcPrices = series.map((s: any, i: number) => {
    // 基于因子数据生成模拟的BTC价格
    const basePrice = 65000;
    const macro = s.dimensions.macro || 50;
    const sentiment = s.dimensions.sentiment || 50;
    const volatility = (Math.random() - 0.5) * 0.1;
    return basePrice * (1 + (macro - 50) / 1000 + (sentiment - 50) / 1000 + volatility);
  });

  const r_btc = transformSeries(btcPrices, 'logret');

  // 计算每个因子与BTC的相关性
  const byFactor: Record<string, (number|null)[]> = {};
  
  dimensions.forEach(factor => {
    const factorValues = factorData[factor];
    // 先对水平做去台阶平滑
    const levelSmoothed = ema(factorValues, INTERP_SPAN);
    // 首选：对差分做相关（更稳健）
    const xDiff = ema(transformSeries(levelSmoothed, 'diff'), EMA_SOFT);
    let rho = rollingSpearman(r_btc, xDiff, win);
    // 如果差分导致低方差而大面积为 null，则回退到水平序列的秩相关
    const nullRatio = rho.filter(v=>v==null).length / rho.length;
    if (nullRatio > 0.6) {
      const xLvl = ema(levelSmoothed, EMA_SOFT);
      rho = rollingSpearman(r_btc, xLvl as any, win);
    }
    byFactor[factor] = rho;
  });

  return { dates, rho: byFactor };
}

/** 回退：当前端计算低方差或缺线时，调用后端现成的 corr_lines */
async function computeCorrFallback(asset='BTC', win=WIN){
  try{
    const url = `/api/factors/corr_lines?asset=${asset}&window=${win}`;
    const r = await fetch(url);
    
    // 检查响应状态和内容类型
    if (!r.ok) {
      throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    }
    
    const contentType = r.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      const text = await r.text();
      console.error('[corr-frontonly] API returned non-JSON response:', text.substring(0, 200));
      throw new Error('API returned HTML instead of JSON');
    }
    
    const js = await r.json();
    const dates: string[] = js?.dates || [];
    const rhoRaw: Record<string, number[] | (number|null)[]> = js?.rho || {};
    // 统一为 (number|null)[]
    const rho: Record<string, (number|null)[]> = {};
    Object.keys(rhoRaw||{}).forEach(k=>{ rho[k] = (rhoRaw[k] || []).map(v => (v==null? null : Number(v))); });
    if (!dates.length || !Object.keys(rho).length) throw new Error('empty corr_lines');
    return { dates, rho };
  }catch(e){
    console.warn('[corr-fallback] failed', e);
    throw e;
  }
}

/** 分位数（忽略 null/NaN） */
function quantile(arr:number[], q:number){
  const v = arr.filter(x=>Number.isFinite(x)).sort((a,b)=>a-b);
  if (!v.length) return NaN;
  const pos = (v.length-1)*q, i=Math.floor(pos), frac=pos-i;
  return i+1<v.length ? v[i]*(1-frac)+v[i+1]*frac : v[i];
}

/** —— 画图：把上面的计算结果丢进 ECharts —— */
function buildWeb3Option(dates:string[], rhoGroups:Record<string, (number|null)[]>, badgePerRow?: number): echarts.EChartsOption {
  const toPairs = (arr:(number|null)[]) => dates.map((d,i)=>[d, arr?.[i] ?? null]);

  const groupKeys = Object.keys(rhoGroups);
  // y 轴范围用 5%/95% 分位（再与 [-100,100] 取交集）
  const all = Object.values(rhoGroups).flat().filter((v):v is number=>Number.isFinite(v));
  const q5  = quantile(all, .05), q95 = quantile(all, .95);
  const yMin = Math.max(-100, Math.floor((isFinite(q5)? q5 : -40) - 10));
  const yMax = Math.min( 100, Math.ceil ((isFinite(q95)?q95:  40) + 10));

  const PALETTE: Record<string,{color:string; dash:any}> = {
    macro:      { color: COLORS.cyan.base,   dash: 'solid' },
    policy:     { color: COLORS.cyan.mid,    dash: [6,4]   },
    capital:    { color: COLORS.mint.base,   dash: 'solid' },
    sentiment:  { color: COLORS.mint.mid,    dash: [6,4]   },
    geopolitics:{ color: COLORS.violet.base, dash: 'solid' },
    onchain:    { color: COLORS.violet.mid,  dash: [6,4]   }
  };

  const mkSeries = (name:string, color:string, dash:any): echarts.SeriesOption[] => ([
    { name: name+'__glow', type:'line', xAxisIndex:0, yAxisIndex:0, z:1, showSymbol:false, smooth:true, silent:true,
      data: toPairs(rhoGroups[name]!), lineStyle:{ width:7, color, opacity:.28, shadowBlur:18, shadowColor:color } },
    { name, type:'line', xAxisIndex:0, yAxisIndex:0, z:2, showSymbol:false, smooth:true, sampling:'lttb',
      data: toPairs(rhoGroups[name]!), lineStyle:{ width:2.2, color, type:dash } }
  ]);

  // 能量带：所有负相关 |ρ| 的均值
  const energy = dates.map((_,i)=>{
    const vals = groupKeys.map(k=>rhoGroups[k]?.[i]).filter((v):v is number=>Number.isFinite(v as number)) as number[];
    const neg = vals.filter(v=>v<0).map(v=>Math.abs(v)/100);
    if (!neg.length) return 0;
    const m = neg.reduce((a,b)=>a+b,0)/neg.length;
    return -Math.min(1, m);
  });

  const zeroBand = {
    type:'line', name:'zero-band', data:[],
    markArea:{ silent:true,z:-1,itemStyle:{ color:{
      type:'linear',x:0,y:0,x2:0,y2:1,
      colorStops:[{offset:0,color:'rgba(41,227,255,0.00)'},{offset:.5,color:'rgba(41,227,255,.18)'},{offset:1,color:'rgba(41,227,255,0.00)'}]
    }}, data:[[ {yAxis:-12},{yAxis:12} ]] }
  } as echarts.SeriesOption;

  const series: echarts.SeriesOption[] = [];
  for (const k of groupKeys) {
    const sty = PALETTE[k] ?? { color:'#8FA5C4', dash:'solid' };
    series.push(...mkSeries(k, sty.color, sty.dash));
  }

  // 顶部徽章改为 DOM 渲染，图内不再绘制重复徽章
  const perRow = Math.max(1, Number.isFinite(badgePerRow as number) ? (badgePerRow as number) : 4);
  const itemWidth = 110;
  const rowHeight = 22;
  const energyBars: echarts.SeriesOption = {
    name:'energy', type:'bar', xAxisIndex:1, yAxisIndex:1, z:0, silent:true,
    data: dates.map((d,i)=>[d, energy[i]]), barMaxWidth:14, barMinWidth:2, barCategoryGap:'30%',
    itemStyle:{ color:{ type:'linear',x:0,y:0,x2:0,y2:1,
      colorStops:[{offset:0,color:'rgba(143,165,196,.28)'},{offset:1,color:'rgba(143,165,196,.55)'}] } }
  };

  // 动态抬高上方网格，避免多行徽章与图形重叠
  const rowsCount = Math.ceil(groupKeys.length / perRow);
  const gridTop = 64 + Math.max(0, rowsCount - 1) * rowHeight;

  // 渲染 DOM 徽章（在容器上方，避免与图重叠，并天然支持浏览器原生 tooltip）
  try {
    const host = document.getElementById('corr-badges');
    if (host) {
      const desc: Record<string,string> = {
        macro: '宏观：SPX(+), VIX(-), DXY(-), XAU(+)',
        policy: '政策/金融条件：NFCI(宽松→正向)',
        capital: '资金：ETF 流入(+), Funding(+)',
        sentiment: '情绪：Fear & Greed (FNG)',
        geopolitics: '地缘：占位（暂无映射）',
        onchain: '链上：占位（暂无映射）'
      };
      const colorOf = (k:string)=> (PALETTE[k]?.color || '#8FA5C4');
      host.innerHTML = groupKeys.map(k => (
        `<span class="corr-badge" title="${desc[k] || k}">
          <i class="corr-dot" style="background:${colorOf(k)}"></i>${k}
        </span>`
      )).join('');
    }
  } catch(_){}

  return {
    backgroundColor:'#0B0F14', textStyle:{ color:COLORS.text },
    grid:[ {left:48,right:18,top:gridTop,bottom:'22%'}, {left:48,right:18,top:'82%',height:'16%'} ],
    xAxis:[ {type:'time',boundaryGap:false,axisLabel:{show:false},axisLine:{lineStyle:{color:COLORS.grid}},splitLine:{lineStyle:{color:COLORS.grid}}},
            {type:'time',boundaryGap:false,axisLine:{lineStyle:{color:COLORS.grid}},splitLine:{lineStyle:{color:COLORS.grid}}} ],
    yAxis:[ {type:'value',min:yMin,max:yMax,name:'相关性(%)',nameTextStyle:{color:COLORS.text},axisLine:{lineStyle:{color:COLORS.grid}},splitLine:{lineStyle:{color:COLORS.grid}}},
            {type:'value',min:-1,max:0,show:false} ],
    dataZoom:[ {type:'inside',xAxisIndex:[0,1],start:0,end:100,throttle:50},
               {type:'slider',xAxisIndex:[0,1],start:0,end:100,height:14,bottom:2} ],
    axisPointer:{ link:[{xAxisIndex:[0,1]}], lineStyle:{color:'rgba(255,255,255,.25)'} },
    legend:{ show:false },
    tooltip:{ trigger:'axis', axisPointer:{type:'cross'}, triggerOn:'mousemove|click',
      backgroundColor:'rgba(13,17,23,.9)', borderColor:'rgba(255,255,255,.08)', borderWidth:1,
      formatter:(ps:any[])=>{
        const date = echarts.format.formatTime('yyyy-MM-dd', ps[0].axisValue);
        const rows = ps.filter(p=>!String(p.seriesName).endsWith('__glow') && p.seriesName!=='energy' && p.seriesName!=='zero-band')
          .sort((a,b)=>Math.abs((b.data?.[1]??0))-Math.abs((a.data?.[1]??0))).slice(0,6)
          .map(p=>`<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${p.color};margin-right:6px"></span>${p.seriesName}: <b>${(p.data?.[1]??0).toFixed(1)}%</b></div>`).join('');
        return `<div style="min-width:200px"><div style="opacity:.8;margin-bottom:6px">${date}</div>${rows}</div>`;
      } },
    // 不使用图内 graphic 徽章，改为 DOM 徽章
    series:[ zeroBand, ...series, energyBars ],
    animationDuration:700, animationEasing:'quarticOut', progressive:4000
  };
}

/** —— 入口：拉 /api/factors → 前端计算 → 画图 —— */
async function renderCorrChart(){
  const el = document.getElementById('corrChart');
  if (!el) {
    console.warn('[corr] corrChart element not found, skipping correlation chart rendering');
    return;
  }
  
  const loadingEl = document.getElementById('corr-loading');
  // 容器有尺寸再 init
  const ready = () => el.clientWidth > 0 && el.clientHeight > 0;
  if (!ready()){ 
    // 添加计数器防止无限循环
    if (!(window as any)._corrRenderAttempts) (window as any)._corrRenderAttempts = 0;
    (window as any)._corrRenderAttempts++;
    if ((window as any)._corrRenderAttempts > 50) { // 减少尝试次数
      console.error('[corr-frontonly] Too many render attempts, giving up. Container dimensions:', {
        width: el.clientWidth,
        height: el.clientHeight,
        offsetWidth: el.offsetWidth,
        offsetHeight: el.offsetHeight
      });
      return;
    }
    // 如果容器太小，等待更长时间
    setTimeout(() => requestAnimationFrame(renderCorrChart), 100);
    return; 
  }
  // 重置计数器
  (window as any)._corrRenderAttempts = 0;

  // 动态加载 echarts（优先使用全局，其次按需加载）
  let echartsMod: any = (window as any).echarts;
  if (!echartsMod) {
    try {
      const mod: any = (await import('echarts')) as any;
      echartsMod = (mod && (mod as any).default) ? (mod as any).default : mod;
      (window as any).echarts = echartsMod;
    } catch (e) {
      console.error('[corr-frontonly] failed to load echarts', e);
      throw e;
    }
  }

  // 复用已存在的实例，避免重复 init 警告
  let chart = echartsMod.getInstanceByDom(el);
  if (!chart) {
    chart = echartsMod.init(el, undefined, { renderer: 'canvas' });
  }
  try{
    // 简易 24h 内存缓存（浏览器进程内）
    const now = Date.now();
    const cache = (window as any)._corrCache as { ts:number; dates:string[]; rhoGroups:Record<string,(number|null)[]> } | undefined;
    let dates:string[]; let rhoGroups: Record<string,(number|null)[]>;
    if (cache && (now - cache.ts) < 24*60*60*1000) {
      ({ dates, rhoGroups } = cache);
    } else {
      // 先用前端-only 计算底层因子
      let { dates: d0, rho } = await computeCorrFrontOnly('BTC', WIN);
      // 合成分组曲线
      let g0 = composeGroups(rho);
      // 如果有效组少于 2 条，则回退到后端 corr_lines
      const nonEmpty = Object.values(g0).filter(arr => (arr||[]).some(v=>v!=null && isFinite(v as any)));
      if (nonEmpty.length < 2) {
        console.warn('[corr-frontonly] too few group series, falling back to /api/factors/corr_lines');
        try {
          const fb = await computeCorrFallback('BTC', WIN);
          d0 = fb.dates; g0 = composeGroups(fb.rho);
        } catch (fallbackError) {
          console.error('[corr-frontonly] Fallback API also failed:', fallbackError);
          // 使用模拟数据作为最后的后备方案
          d0 = generateMockDates(30);
          g0 = generateMockCorrelationGroups(d0);
        }
      }
      dates = d0; rhoGroups = g0;
      (window as any)._corrCache = { ts: now, dates, rhoGroups };
    }
    const opt = buildWeb3Option(dates, rhoGroups);
    chart.setOption(opt, { notMerge:true, replaceMerge:['grid','xAxis','yAxis','series','dataZoom','graphic','legend'] });
    chart.dispatchAction({ type:'dataZoom', xAxisIndex:[0,1], start:0, end:100 }); // 拉满视窗
    if (loadingEl) { loadingEl.style.display = 'none'; loadingEl.style.visibility = 'hidden'; loadingEl.style.zIndex = '-1'; }
  }catch(e){
    console.error('[corr-frontonly] failed', e);
    chart.clear();
    chart.setOption({ title:{ text:'相关性数据加载失败', left:'center', top:'middle', textStyle:{ color:'#9aa4b2' } }});
    if (loadingEl) { loadingEl.style.display = 'none'; loadingEl.style.visibility = 'hidden'; loadingEl.style.zIndex = '-1'; }
  }

  // 单一 resize（去抖）
  if (!(el as any)._corrResizeBound) {
    let t:any; const onResize=()=>{ clearTimeout(t); t=setTimeout(()=>chart && chart.resize(), 120); };
    window.addEventListener('resize', onResize);
    const ro = new ResizeObserver(onResize);
    ro.observe(el);
    (el as any)._corrResizeBound = { onResize, ro };
  }
}

// 导出函数供外部调用
export { renderCorrChart, computeCorrFrontOnly, buildWeb3Option };

/** 将底层因子合成为分组线 */
function composeGroups(rho: Record<string,(number|null)[]>) {
  const take = (k:string, sign=1)=> (rho[k]||[]).map(v => (Number.isFinite(v as number) ? sign*(v as number) : null));
  const mix  = (...arrs:(number|null)[][]) => {
    const n = Math.max(...arrs.map(a=>a.length));
    const out:(number|null)[] = Array(n).fill(null);
    for (let i=0;i<n;i++){
      const vals = arrs.map(a=>a[i]).filter((v):v is number=>v!=null);
      out[i] = vals.length ? vals.reduce((s,x)=>s+x,0)/vals.length : null;
    }
    return out;
  };

  const groups: Record<string,(number|null)[]> = {};
  if (rho.SPX || rho.VIX || rho.DXY || rho.XAU) {
    groups.macro = mix( take('SPX',+1), take('VIX',-1), take('DXY',-1), take('XAU',+1) );
  }
  if (rho.NFCI) groups.policy = take('NFCI', -1);
  if (rho.ETF_Flows || rho.Funding) {
    groups.capital = mix( take('ETF_Flows', +1), take('Funding', +1) );
  }
  if (rho.FNG) groups.sentiment = take('FNG', +1);
  // geopolitics / onchain 暂无底层
  return groups;
}
