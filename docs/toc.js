// Section list left of the text, built from the h2/h3 in <main>. Starts level with the first
// heading and sticks to the top of the window from there (position: sticky). The current section
// is the last heading above 30% of the window height.
const main = document.querySelector("main");
const heads = [...main.querySelectorAll("h2, h3")];
const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
for (const h of heads) if (!h.id) h.id = slug(h.textContent);

const nav = document.createElement("nav");
nav.className = "toc";
nav.setAttribute("aria-label", "Contents");
const head = document.createElement("p");
head.className = "head";
head.innerHTML = `<a href="#">${document.querySelector("h1").textContent}</a>`;
head.firstChild.addEventListener("click", (e) => { e.preventDefault(); scrollTo({ top: 0 }); });
nav.appendChild(head);
const ol = document.createElement("ol");
const mark = document.createElement("span");
mark.className = "mark";
ol.appendChild(mark);
const links = heads.map((h) => {
  const li = document.createElement("li");
  li.className = h.tagName === "H3" ? "sub" : "";
  const a = document.createElement("a");
  a.href = `#${h.id}`;
  a.textContent = h.textContent;
  li.appendChild(a);
  ol.appendChild(li);
  return a;
});
nav.appendChild(ol);
const rail = document.createElement("div");
rail.className = "toc-rail";
rail.appendChild(nav);
main.prepend(rail);

// line the top of the sidebar title's text up with the first heading's. The two use different font
// sizes, so their line boxes don't line up. For each, find the baseline with a zero-size probe, then
// draw the text at its real size (the serif changes shape with size) 8x oversampled and find the
// first row of ink above the baseline.
const ctx = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
function ascent(el) {
  const cs = getComputedStyle(el), S = 8, text = el.textContent.trim();
  const font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 4, h = Math.ceil(parseFloat(cs.fontSize) * 2);
  ctx.canvas.width = w * S;
  ctx.canvas.height = h * S;
  ctx.setTransform(S, 0, 0, S, 0, 0);
  ctx.font = font;
  ctx.fillText(text, 2, h);
  const px = ctx.getImageData(0, 0, w * S, h * S).data;
  for (let y = 0; y < h * S; y++)
    for (let x = 0; x < w * S; x++) if (px[4 * (y * w * S + x) + 3] > 64) return h - y / S;
  return 0;
}
function inkTop(el) {
  const probe = document.createElement("span");
  probe.style.cssText = "display:inline-block;width:0;height:0;vertical-align:baseline";
  el.prepend(probe);
  const baseline = probe.getBoundingClientRect().bottom - el.getBoundingClientRect().top;
  probe.remove();
  return baseline - ascent(el);
}
const align = () => { rail.style.top = `${heads[0].offsetTop + inkTop(heads[0]) - inkTop(head)}px`; };
new ResizeObserver(align).observe(main);
document.fonts.ready.then(align);
align();

let cur = -1;
function spy() {
  const y = innerHeight * 0.3;
  let k = 0;
  heads.forEach((h, i) => { if (h.getBoundingClientRect().top < y) k = i; });
  // at the very bottom the last short sections can't reach the line
  if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2) k = heads.length - 1;
  if (k === cur) return;
  cur = k;
  links.forEach((a, i) => a.toggleAttribute("aria-current", i === k));
  place();
}
// the bar on the rule slides to the current entry
function place() {
  const a = links[cur];
  mark.style.transform = `translateY(${a.offsetTop}px)`;
  mark.style.height = `${a.offsetHeight}px`;
}
addEventListener("scroll", spy, { passive: true });
addEventListener("resize", () => { spy(); place(); });
spy();
