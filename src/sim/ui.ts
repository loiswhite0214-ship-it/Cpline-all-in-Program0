// Badge + fly animation utilities (robust to DOM structure changes)

const BADGE_ID = "mine-badge";

function getMineEl(): HTMLElement | null {
  // 优先飞到红点；没有红点就飞到“我的”按钮
  return (document.getElementById('mine-badge') as HTMLElement | null)
      ?? (document.getElementById('nav-mine') as HTMLElement | null);
}

export function updateBadge() {
  const badge = document.getElementById(BADGE_ID) as HTMLSpanElement | null;
  if (!badge) return;
  let n = 0;
  try { n = JSON.parse(localStorage.getItem('simQueue') || '[]').length; } catch { n = 0; }
  if (n > 0) { badge.style.display = "inline-flex"; badge.textContent = String(n); }
  else { badge.style.display = "none"; }
}

export function bumpMinePulse() {
  const mine = getMineEl();
  if (!mine) return;
  mine.classList.add("pulse");
  setTimeout(() => mine.classList.remove("pulse"), 300);
}

export function flyToMine(fromEl: HTMLElement) {
  const mine = getMineEl();
  if (!mine) return;

  // 起点/终点都用 viewport 坐标，配合 position:fixed —— 不受滚动影响
  const s = fromEl.getBoundingClientRect();
  const e = mine.getBoundingClientRect();

  const startX = s.left + s.width / 2;
  const startY = s.top + s.height / 2;
  // 落点用徽标右上角（内缩20%）
  const endX = e.left + e.width / 2;
  const endY = e.top + e.height / 2;

  const dot = document.createElement('div');
  Object.assign(dot.style, {
    position: 'fixed',
    left: `${startX}px`,
    top: `${startY}px`,
    width: '12px',
    height: '12px',
    borderRadius: '999px',
    background: '#0ea5e9',
    zIndex: '2147483647',
    pointerEvents: 'none',
  } as CSSStyleDeclaration);
  document.body.appendChild(dot);

  const dx = endX - startX;
  const dy = endY - startY;

  const anim = dot.animate(
    [
      { transform: 'translate(0,0) scale(1)', opacity: 1 },
      { transform: `translate(${dx*0.55}px, ${dy*0.25}px) scale(1.1)`, opacity: 1, offset: 0.6 },
      { transform: `translate(${dx}px, ${dy}px) scale(0.3)`, opacity: 0.15 },
    ],
    { duration: 650, easing: 'cubic-bezier(.2,.7,.2,1)' }
  );
  anim.onfinish = () => {
    dot.remove();
    const mineBtn = document.getElementById('nav-mine');
    if (mineBtn) {
      mineBtn.classList.add('pulse');
      setTimeout(() => mineBtn.classList.remove('pulse'), 300);
    }
  };
}

