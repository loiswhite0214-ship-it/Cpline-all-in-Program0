import { getQueue, setQueue, toggleQueueDisabled, removeFromQueue,
         getPositions, setPositions, openPositionFromSim, closePosition } from "./sim/store";
import { updateBadge } from "./sim/ui";
import type { SimItem, SimPosition } from "./sim/types";

// ===== 价格适配：可接入事件总线或轮询 =====
const lastPriceMap = new Map<string, number>();
function onPrice(symbol: string, price: number) {
  lastPriceMap.set(symbol, price);
  const list = getPositions().map(p => {
    if (p.status !== "open" || p.symbol !== symbol) return p;
    const dir = p.side === "BUY" ? 1 : -1;
    const pnl = (price - p.avgEntry) * dir * p.qty;
    return { ...p, lastPrice: price, pnl, pnlPct: (pnl / (p.avgEntry * p.qty)) * 100 };
  });
  setPositions(list);
  render();
}

// ============ DOM 辅助 ============
function fmt(t: number) { return new Date(t).toLocaleString(); }
function n(v: any, d = 4) { const x = Number(v); return isNaN(x) ? "-" : x.toFixed(d); }

function render() {
  const elQueued  = document.getElementById("mine-queued");
  const elOpen    = document.getElementById("mine-open");
  const elHistory = document.getElementById("mine-history");
  if (!elQueued || !elOpen || !elHistory) return;

  // 队列
  const q = getQueue();
  elQueued.innerHTML = q.length ? q.map(i => `
    <div class="signal-card ${i.side==='SELL'?'sell':''}" data-simid="${i.id}">
      <div class="signal-header">
        <div class="signal-title">${i.symbol} · ${i.side} · ${i.tf}</div>
        <div class="signal-strategy">${i.strategy}${i.disabled?'｜已停止后续':''}</div>
      </div>
      <div class="signal-details">
        <div>入队：${fmt(i.createdAt)}</div>
        <div>参考入场：${i.entry ?? '-'}</div>
      </div>
      <div style="display:flex; gap:8px; margin-top:10px;">
        <button class="timeframe-btn" data-action="enable" ${i.disabled?'disabled':''}>启用</button>
        <button class="timeframe-btn" data-action="toggle">${i.disabled?'恢复后续启用':'停止后续启用'}</button>
        <button class="timeframe-btn" data-action="remove">移出</button>
      </div>
    </div>
  `).join("") : `<div class="signal-card"><div class="signal-title">队列为空</div></div>`;

  // 运行中
  const pos = getPositions();
  const running = pos.filter(p => p.status === "open");
  elOpen.innerHTML = running.length ? running.map(p => `
    <div class="signal-card ${p.side==='SELL'?'sell':''}" data-posid="${p.posId}" data-from="${p.fromSimId}">
      <div class="signal-header">
        <div class="signal-title">${p.symbol} · ${p.side} · ${p.tf}</div>
        <div class="signal-strategy">${p.strategy}</div>
      </div>
      <div class="signal-details">
        <div>数量：${p.qty}</div>
        <div>均价：${n(p.avgEntry)}</div>
        <div>现价：${p.lastPrice? n(p.lastPrice): '-'}</div>
        <div>浮盈亏：${p.pnl? n(p.pnl,2): '-'}</div>
        <div>浮盈亏%：${p.pnlPct? n(p.pnlPct,2)+'%': '-'}</div>
        <div>开仓：${fmt(p.openTime)}</div>
      </div>
      <div style="display:flex; gap:8px; margin-top:10px;">
        <button class="timeframe-btn" data-action="close">关闭仓位</button>
      </div>
    </div>
  `).join("") : `<div class="signal-card"><div class="signal-title">暂无运行中的模拟仓位</div></div>`;

  // 历史
  const hist = pos.filter(p => p.status === "closed");
  elHistory.innerHTML = hist.length ? `
    <table class="quotes-table">
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>TF</th>
        <th>开仓</th><th>平仓</th><th>盈亏</th><th>盈亏%</th>
      </tr></thead>
      <tbody>
      ${hist.map(p => `
        <tr>
          <td>${p.symbol}</td>
          <td>${p.side}</td>
          <td>${p.tf}</td>
          <td>${fmt(p.openTime)}</td>
          <td>${fmt(p.closeTime!)}</td>
          <td>${p.pnlFinal!==undefined? n(p.pnlFinal,2): '-'}</td>
          <td>${p.pnlFinal!==undefined? n((p.pnlFinal/(p.avgEntry*p.qty))*100,2)+'%': '-'}</td>
        </tr>
      `).join("")}
      </tbody>
    </table>` : `<div class="signal-card"><div class="signal-title">暂无历史</div></div>`;

  updateBadge();
}

// 事件绑定：队列/运行中操作（事件委托）
document.addEventListener("click", (ev) => {
  const btn = (ev.target as HTMLElement).closest("[data-action]") as HTMLElement | null;
  if (!btn) return;

  const action = btn.getAttribute("data-action")!;
  // 队列动作
  const wrapQ = btn.closest("[data-simid]") as HTMLElement | null;
  if (wrapQ) {
    const id = wrapQ.getAttribute("data-simid")!;
    const q = getQueue();
    const item = q.find(i => i.id === id)!;

    if (action === "enable") {
      const price = lastPriceMap.get(item.symbol) ?? Number(item.entry) ?? 0;
      openPositionFromSim(item, price, 1);
      render();
      return;
    }
    if (action === "toggle") {
      toggleQueueDisabled(id, !item.disabled);
      render(); return;
    }
    if (action === "remove") {
      removeFromQueue(id);
      render(); return;
    }
  }

  // 运行中动作
  const wrapP = btn.closest("[data-posid]") as HTMLElement | null;
  if (wrapP) {
    const posId = wrapP.getAttribute("data-posid")!;
    if (action === "close") {
      const also = confirm("是否同时停止该信号的后续启用？\n（确定=停止后续；取消=仅平仓）");
      const price = (() => {
        const p = getPositions().find(p => p.posId === posId)!;
        return lastPriceMap.get(p.symbol) ?? p.avgEntry;
      })();
      closePosition(posId, price, also);
      render(); return;
    }
  }
});

// 一键启用全部
function bindEnableAll() {
  const btnEnableAll = document.getElementById("btn-enable-all");
  if (btnEnableAll) {
    // 移除之前的事件监听器（如果有的话）
    btnEnableAll.replaceWith(btnEnableAll.cloneNode(true));
    const newBtn = document.getElementById("btn-enable-all");
    newBtn?.addEventListener("click", () => {
      const q = getQueue().filter(i => !i.disabled);
      if (q.length === 0) {
        alert("没有可启用的信号");
        return;
      }
      const confirmed = confirm(`确定要启用全部 ${q.length} 个信号吗？`);
      if (confirmed) {
        q.forEach(i => {
          const price = lastPriceMap.get(i.symbol) ?? Number(i.entry) ?? 0;
          openPositionFromSim(i, price, 1);
        });
        render();
        alert(`已启用 ${q.length} 个信号`);
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  render();
  bindEnableAll();
  // 可选：接上你的报价源
  // (window as any).quotesBus?.on("price", ({symbol, price}: any) => onPrice(symbol, price));
});

// 导出供外部可选调用
export { onPrice };

export function initMineUI() {
  try {
    render();
    bindEnableAll();
  } catch (_) {}
}

