// The scrolling story on the landing page. The scroll position scrubs one
// pinned drawing through a sequence of real solves, stored by
// docs/landing_data.py in data.json: field lines while the plate's
// susceptibility rises, while the magnet lifts off, and how each solver cuts
// the plate.
(() => {
  const root = document.getElementById("mrl");
  if (!root) return;
  // the data sits next to the script, or in the page itself when the story
  // is served as one file
  const script = document.currentScript;
  const load = () =>
    window.MRL_DATA
      ? Promise.resolve(window.MRL_DATA)
      : fetch(new URL("data.json", script.src)).then((r) => r.json());
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const stage = root.querySelector(".mrl-stage");
  const canvas = stage.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const beats = [...root.querySelectorAll(".mrl-beat")];
  const hud = Object.fromEntries(
    [...root.querySelectorAll("[data-hud]")].map((el) => [el.dataset.hud, el]),
  );
  const css = getComputedStyle(root);
  const colour = (name) => css.getPropertyValue(name).trim();
  const C = {
    north: colour("--mrl-north"),
    south: colour("--mrl-south"),
    line: colour("--mrl-line"),
    fc: colour("--mrl-fc"),
    cells: colour("--mrl-cells"),
    fem: colour("--mrl-fem"),
  };

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (a, b, x) => {
    const t = clamp((x - a) / (b - a));
    return t * t * (3 - 2 * t);
  };
  const fmt = (x, digits = 0) =>
    x.toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });

  // the story fills the window's width, out of the theme's column, and
  // starts right under its header
  const fitWidth = () => {
    root.style.marginLeft = "0px";
    root.style.marginTop = "0px";
    const r = root.getBoundingClientRect();
    const main = root.closest(".bd-main") || document.body;
    root.style.width = `${document.documentElement.clientWidth}px`;
    root.style.marginLeft = `${-r.left}px`;
    root.style.marginTop = `${-(r.top - main.getBoundingClientRect().top)}px`;
  };

  let data = null;
  let size = { w: 0, h: 0, dpr: 1 };

  function resize() {
    fitWidth();
    const r = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    size = { w: r.width, h: r.height, dpr };
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    draw();
  }

  // ── the data ────────────────────────────────────────────────────────────

  function decode(frame) {
    return frame.lines.map((line) => ({
      xz: Float32Array.from(line.xz, (v) => v / 100),
      b: Float32Array.from(line.b, (v) => v / 99),
      seed: line.seed,
      closed: line.closed,
    }));
  }

  // the lines between two stored frames, point by point
  function blend(frames, u) {
    const i = clamp(Math.floor(u), 0, frames.length - 2);
    const t = clamp(u - i);
    const a = frames[i].lines;
    const b = frames[i + 1].lines;
    return a.map((la, k) => {
      const lb = b[k];
      const xz = new Float32Array(la.xz.length);
      const s = new Float32Array(la.b.length);
      for (let j = 0; j < xz.length; j++) xz[j] = lerp(la.xz[j], lb.xz[j], t);
      for (let j = 0; j < s.length; j++) s[j] = lerp(la.b[j], lb.b[j], t);
      return { xz, b: s, seed: la.seed, closed: la.closed };
    });
  }

  const value = (frames, u, key, log = false) => {
    const i = clamp(Math.floor(u), 0, frames.length - 2);
    const t = clamp(u - i);
    const a = frames[i][key];
    const b = frames[i + 1][key];
    return log ? Math.exp(lerp(Math.log(a), Math.log(b), t)) : lerp(a, b, t);
  };

  // ── where the story is ──────────────────────────────────────────────────

  // the story's clock: beat i runs from i to i + 1 while its block crosses the
  // middle of the window
  function clock() {
    const middle = window.innerHeight / 2;
    let T = 0;
    for (const beat of beats) {
      const r = beat.getBoundingClientRect();
      T += clamp((middle - r.top) / r.height);
    }
    return T;
  }

  function scene(T) {
    const s = {};
    s.layout = smooth(0.3, 0.95, T); // the drawing moves aside for the words
    s.plate = smooth(1.9, 2.45, T);
    s.chi = (data.chi.length - 1) * smooth(3.0, 3.85, T);
    const up = smooth(5.0, 5.72, T);
    const fall = smooth(5.76, 5.94, T) ** 2;
    s.lift = (data.lift.length - 1) * up * (1 - fall);
    s.lifting = T > 4.95 && T < 6.0;
    s.fc = smooth(6.0, 6.5, T);
    s.fcOut = smooth(6.92, 7.1, T);
    s.cells = smooth(7.0, 7.5, T);
    s.cellsOut = smooth(7.92, 8.1, T);
    s.fem = smooth(8.0, 8.5, T);
    s.solver = Math.max(
      s.fc * (1 - s.fcOut),
      s.cells * (1 - s.cellsOut),
      s.fem,
    );
    s.sensor = smooth(2.6, 2.9, T) * (1 - smooth(5.95, 6.15, T));
    s.labels = smooth(0.8, 1.2, T) * (1 - smooth(2.8, 3.1, T));
    return s;
  }

  // ── drawing ─────────────────────────────────────────────────────────────

  // world (mm) to pixels: the window, in the region of the stage it takes
  function view(layout) {
    const { w, h } = size;
    const [x0, x1, z0, z1] = data.window;
    const wide = w >= 900;
    // at the opening the magnet sits under the words, its lines behind them
    const full = wide
      ? { x: 0, y: h * 0.3, w, h }
      : { x: 0, y: h * 0.62, w, h: h * 0.66 };
    const aside = wide
      ? { x: w * 0.36, y: 0, w: w * 0.64, h }
      : { x: 0, y: 0, w, h: h * 0.56 };
    const R = {
      x: lerp(full.x, aside.x, layout),
      y: lerp(full.y, aside.y, layout),
      w: lerp(full.w, aside.w, layout),
      h: lerp(full.h, aside.h, layout),
    };
    const pad = lerp(0.02, wide ? 0.06 : 0.04, layout);
    const s = Math.min(
      (R.w * (1 - 2 * pad)) / (x1 - x0),
      (R.h * (1 - 2 * pad)) / (z1 - z0),
    );
    const cx = R.x + R.w / 2;
    const cy = R.y + R.h / 2;
    const mx = (x0 + x1) / 2;
    const mz = (z0 + z1) / 2;
    return {
      s,
      X: (x) => cx + (x - mx) * s,
      Z: (z) => cy - (z - mz) * s,
    };
  }

  function rect(v, x, z, w, h) {
    ctx.rect(v.X(x), v.Z(z + h), w * v.s, h * v.s);
  }

  // how far along a line segment j is from where the line leaves the magnet,
  // 0 to 1: the opening grows the lines out of it
  function reach(line, j) {
    const n = line.b.length - 1;
    if (line.closed) return Math.min(j, n - 1 - j) / ((n - 1) / 2);
    const d = j < line.seed ? line.seed - 1 - j : j - line.seed;
    return d / Math.max(line.seed, n - line.seed, 1);
  }

  function fieldLines(v, lines, alpha, t, grow = 1) {
    // the lines, brighter where the field is stronger
    const BINS = 6;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (let bin = 0; bin < BINS; bin++) {
      const path = new Path2D();
      for (const line of lines) {
        const { xz, b } = line;
        for (let j = 0; j + 1 < b.length; j++) {
          if (grow < 1 && reach(line, j) > grow) continue;
          const level = Math.min(
            BINS - 1,
            Math.floor(((b[j] + b[j + 1]) / 2) * BINS),
          );
          if (level !== bin) continue;
          path.moveTo(v.X(xz[2 * j]), v.Z(xz[2 * j + 1]));
          path.lineTo(v.X(xz[2 * j + 2]), v.Z(xz[2 * j + 3]));
        }
      }
      const strength = (bin + 0.5) / BINS;
      ctx.strokeStyle = C.line;
      ctx.globalAlpha = alpha * (0.05 + 0.16 * strength);
      ctx.lineWidth = 6;
      ctx.stroke(path);
      ctx.globalAlpha = alpha * (0.18 + 0.75 * strength ** 1.4);
      ctx.lineWidth = 1.3;
      ctx.stroke(path);
    }
    // and the flux running along them, in the direction of B
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.6;
    ctx.globalAlpha = alpha * 0.9 * smooth(0.8, 1, grow);
    ctx.setLineDash([0.1, 38]);
    ctx.lineDashOffset = -t * 0.03;
    for (const { xz } of lines) {
      ctx.beginPath();
      ctx.moveTo(v.X(xz[0]), v.Z(xz[1]));
      for (let j = 2; j < xz.length; j += 2)
        ctx.lineTo(v.X(xz[j]), v.Z(xz[j + 1]));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  function plate(v, show, response) {
    if (show <= 0) return;
    ctx.globalAlpha = show;
    const g = ctx.createLinearGradient(0, v.Z(3), 0, v.Z(-3));
    g.addColorStop(0, `rgba(150, 162, 180, ${0.1 + 0.22 * response})`);
    g.addColorStop(1, `rgba(90, 100, 118, ${0.06 + 0.16 * response})`);
    ctx.fillStyle = g;
    ctx.beginPath();
    rect(v, -20, -3, 40, 6);
    ctx.fill();
    ctx.strokeStyle = "rgba(200, 210, 225, 0.55)";
    ctx.lineWidth = 1;
    ctx.setLineDash(response > 0.02 ? [] : [5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // the plate's poles: its surface charge, north red, south green
  function charges(v, alpha, width, ticks, reveal = 1) {
    if (alpha <= 0) return;
    const segs = data.solvers["face charge"];
    const level = data.sigmaLevel;
    const edge = lerp(-33, 33, reveal);
    ctx.lineCap = "butt";
    const inset = width / 2 / v.s;
    for (const seg of segs) {
      const [sa, sb] = [seg[4], seg[5]];
      if ((seg[0] + seg[2]) / 2 > edge) continue;
      // drawn just inside the plate, so the pole under the magnet shows
      const flat = Math.abs(seg[1] - seg[3]) < 1e-6;
      const dx = flat ? 0 : -Math.sign(seg[0]) * inset;
      const dz = flat ? -Math.sign(seg[1]) * inset : 0;
      const [xa, za, xb, zb] = [
        seg[0] + dx,
        seg[1] + dz,
        seg[2] + dx,
        seg[3] + dz,
      ];
      const g = ctx.createLinearGradient(v.X(xa), v.Z(za), v.X(xb), v.Z(zb));
      for (const [stop, sigma] of [
        [0, sa],
        [1, sb],
      ]) {
        const k = clamp(Math.abs(sigma) / level);
        const c = sigma > 0 ? C.north : C.south;
        g.addColorStop(
          stop,
          c +
            Math.round(90 + 165 * k)
              .toString(16)
              .padStart(2, "0"),
        );
      }
      ctx.strokeStyle = g;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(v.X(xa), v.Z(za));
      ctx.lineTo(v.X(xb), v.Z(zb));
      ctx.stroke();
    }
    if (ticks > 0) {
      // where one panel ends and the next begins
      ctx.globalAlpha = alpha * ticks;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < segs.length; i++) {
        const [xa, za, xb, zb] = segs[i];
        if ((xa + xb) / 2 > edge) continue;
        const horizontal = Math.abs(za - zb) < 1e-6;
        const out = (horizontal ? Math.sign(za) : Math.sign(xa)) || 1;
        const nx = horizontal ? 0 : out;
        const nz = horizontal ? out : 0;
        for (const [x, z] of [
          [xa, za],
          [xb, zb],
        ]) {
          ctx.moveTo(v.X(x) - nx * 5, v.Z(z) + nz * 5);
          ctx.lineTo(v.X(x) + nx * 5, v.Z(z) - nz * 5);
        }
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function cells(v, alpha, reveal) {
    if (alpha <= 0) return;
    const list = data.solvers.cells;
    const edge = lerp(-21, 21, reveal);
    const top = data.cellsMax;
    for (const [x, z, dx, dz, jx, jz] of list) {
      if (x > edge) continue;
      const j = Math.hypot(jx, jz) / top;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = `rgba(90, 169, 255, ${0.08 + 0.5 * j ** 0.7})`;
      ctx.strokeStyle = "rgba(170, 205, 255, 0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      rect(v, x - dx / 2, z - dz / 2, dx, dz);
      ctx.fill();
      ctx.stroke();
      // the cell's polarization
      const L = 0.42 * Math.min(dx, dz) * (0.35 + 0.65 * j ** 0.5);
      const ux = jx / (Math.hypot(jx, jz) || 1);
      const uz = jz / (Math.hypot(jx, jz) || 1);
      const ax = v.X(x - ux * L);
      const az = v.Z(z - uz * L);
      const bx = v.X(x + ux * L);
      const bz = v.Z(z + uz * L);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(ax, az);
      ctx.lineTo(bx, bz);
      const head = Math.min(5, 0.35 * L * v.s);
      const a = Math.atan2(bz - az, bx - ax);
      ctx.lineTo(bx - head * Math.cos(a - 0.5), bz - head * Math.sin(a - 0.5));
      ctx.moveTo(bx, bz);
      ctx.lineTo(bx - head * Math.cos(a + 0.5), bz - head * Math.sin(a + 0.5));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function elements(v, alpha, reveal) {
    if (alpha <= 0) return;
    const edge = lerp(-33, 33, reveal);
    const path = new Path2D();
    for (const poly of data.solvers["finite elements"]) {
      if (poly[0] / 100 > edge) continue;
      path.moveTo(v.X(poly[0] / 100), v.Z(poly[1] / 100));
      for (let j = 2; j < poly.length; j += 2)
        path.lineTo(v.X(poly[j] / 100), v.Z(poly[j + 1] / 100));
      path.closePath();
    }
    ctx.globalAlpha = alpha * 0.55;
    ctx.strokeStyle = C.fem;
    ctx.lineWidth = 0.6;
    ctx.stroke(path);
    ctx.globalAlpha = 1;
  }

  function magnet(v, gap, alpha = 1) {
    const x = -5;
    const z = 3 + gap;
    ctx.globalAlpha = alpha;
    for (const [z0, c] of [
      [z + 2.5, C.north],
      [z, C.south],
    ]) {
      const g = ctx.createLinearGradient(v.X(x), 0, v.X(x + 10), 0);
      g.addColorStop(0, c + "dd");
      g.addColorStop(0.5, c);
      g.addColorStop(1, c + "bb");
      ctx.fillStyle = g;
      ctx.beginPath();
      rect(v, x, z0, 10, 2.5);
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    rect(v, x, z, 10, 5);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function label(v, text, x, z, alpha, align = "left") {
    if (alpha <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "rgba(230, 236, 245, 0.9)";
    ctx.font =
      "500 13px -apple-system, BlinkMacSystemFont, 'Inter', sans-serif";
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(text, v.X(x), v.Z(z));
    ctx.globalAlpha = 1;
  }

  function sensor(v, gap, bz, alpha) {
    if (alpha <= 0) return;
    const x = v.X(0);
    const z = v.Z(8.5 + gap);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, z, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
    ctx.beginPath();
    ctx.arc(x, z, 9, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.font =
      "600 15px -apple-system, BlinkMacSystemFont, 'Inter', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const text = `${fmt(bz)} mT`;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(7, 9, 13, 0.75)";
    ctx.beginPath();
    ctx.roundRect(x + 14, z - 13, tw + 16, 26, 13);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x + 22, z + 1);
    ctx.globalAlpha = 1;
  }

  function pull(v, gap, force, alpha) {
    if (alpha <= 0) return;
    // the force on the magnet, down toward the plate; its length grows as the
    // square root of the force
    const x = v.X(9);
    const top = v.Z(5.5 + gap + 1.5);
    const L = (4 + 11 * Math.sqrt(force / data.lift[0].force)) * v.s;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = "#ffc247";
    ctx.fillStyle = "#ffc247";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + L - 9);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 7, top + L - 10);
    ctx.lineTo(x + 7, top + L - 10);
    ctx.lineTo(x, top + L);
    ctx.closePath();
    ctx.fill();
    ctx.font =
      "600 15px -apple-system, BlinkMacSystemFont, 'Inter', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${fmt(force, force < 10 ? 1 : 0)} N`, x + 12, top + L / 2);
    ctx.globalAlpha = 1;
  }

  let T = 0;
  let now = 0;
  let intro = null; // when the opening started, if the page opened at the top

  function draw() {
    if (!data || !size.w) return;
    const { w, h, dpr } = size;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const s = scene(T);
    const v = view(s.layout);

    // which lines: the susceptibility rising, then the magnet lifting
    let lines;
    let bz;
    let gap = 0;
    if (s.lifting) {
      lines = blend(data.liftLines, s.lift);
      bz = value(data.lift, s.lift, "bz");
      gap = value(data.lift, s.lift, "gap");
    } else {
      lines = blend(data.chiLines, s.chi);
      bz = value(data.chi, s.chi, "bz");
    }
    const response = clamp(
      (bz - data.chi[0].bz) / (data.chi.at(-1).bz - data.chi[0].bz),
    );
    const chi =
      s.chi < 1 ? s.chi * data.chi[1].chi : value(data.chi, s.chi, "chi", true);

    // the drawing sits back behind the opening words
    const presence = lerp(0.5, 1, s.layout);
    elements(v, s.fem, s.fem);
    plate(v, s.plate, s.lifting ? 1 : response);
    cells(v, s.cells * (1 - s.cellsOut), s.cells);
    const grow =
      intro === null ? 1 : 1 - (1 - clamp((now - intro) / 2400)) ** 3;
    fieldLines(v, lines, presence * (1 - 0.72 * s.solver), now, grow);
    const poles =
      s.plate * response * (s.lifting ? 1 - smooth(0, 0.6, s.lift) : 1);
    const fcShow = s.fc * (1 - s.fcOut);
    charges(
      v,
      Math.max(poles * 0.85, fcShow),
      lerp(3, 6, fcShow),
      fcShow,
      fcShow > 0 ? s.fc : 1,
    );
    magnet(v, gap, presence);
    label(v, "magnet", -6.5, 5.5, s.labels, "right");
    label(v, "steel plate", -19, -5.2, s.labels * s.plate);
    sensor(v, gap, bz, s.sensor);
    const force = value(data.lift, s.lift, "force", true);
    pull(
      v,
      gap,
      force,
      s.lifting ? smooth(4.95, 5.15, T) * (1 - smooth(5.85, 6.0, T)) : 0,
    );

    // the read-outs
    hud.chi.classList.toggle("on", T > 2.85 && T < 5.0);
    hud.bz.classList.toggle("on", s.sensor > 0.5);
    hud.gap.classList.toggle("on", s.lifting && T < 5.85);
    hud.force.classList.toggle("on", s.lifting && T < 5.85);
    hud.chi.querySelector("b").textContent = chi < 1 ? fmt(chi, 1) : fmt(chi);
    hud.bz.querySelector("b").textContent = `${fmt(bz)} mT`;
    hud.gap.querySelector("b").textContent = `${fmt(gap, 1)} mm`;
    hud.force.querySelector("b").textContent =
      `${fmt(force, force < 10 ? 1 : 0)} N`;
  }

  // the words: each beat comes up as it nears the middle of the window
  function words() {
    const vh = window.innerHeight;
    for (const beat of beats) {
      const r = beat.getBoundingClientRect();
      const d = (r.top + r.height / 2 - vh / 2) / vh;
      const o = 1 - smooth(0.16, 0.4, Math.abs(d));
      beat.style.opacity = o.toFixed(3);
      beat.style.transform = reduced
        ? ""
        : `translateY(${(d * 40).toFixed(1)}px)`;
    }
    const hint = root.querySelector(".mrl-hint");
    hint.style.opacity = String(1 - smooth(0.05, 0.3, T));
    // the gain counts up as its beat comes in
    const gain = root.querySelector('[data-fill="gain"]');
    if (gain && data) {
      const k = smooth(3.85, 4.35, T);
      gain.textContent = `+${fmt(k * data.gain)}`;
    }
  }

  // ── after the stage ─────────────────────────────────────────────────────

  function chart() {
    const svg = root.querySelector(".mrl-chart");
    const NS = "http://www.w3.org/2000/svg";
    const a = data.axis;
    // a narrower drawing on a phone, so that its words stay readable
    const narrow = root.clientWidth < 600;
    const W = narrow ? 420 : 800;
    const H = narrow ? 340 : 400;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const m = { l: 44, r: narrow ? 104 : 150, t: 20, b: 46 };
    const zs = a.z;
    const all = [...a["magnet alone"], ...a["finite elements"]];
    const lo = Math.log10(Math.min(...all) * 0.9);
    const hi = Math.log10(Math.max(...all) * 1.1);
    const X = (z) =>
      m.l + ((z - zs[0]) / (zs.at(-1) - zs[0])) * (W - m.l - m.r);
    const Y = (b) => m.t + ((hi - Math.log10(b)) / (hi - lo)) * (H - m.t - m.b);
    const el = (name, attrs, parent = svg) => {
      const e = document.createElementNS(NS, name);
      for (const [k, val] of Object.entries(attrs)) e.setAttribute(k, val);
      parent.appendChild(e);
      return e;
    };
    // the grid
    for (const b of [20, 50, 100, 200, 500]) {
      if (Math.log10(b) < lo || Math.log10(b) > hi) continue;
      el("line", {
        x1: m.l,
        x2: W - m.r,
        y1: Y(b),
        y2: Y(b),
        stroke: "rgba(255,255,255,0.08)",
      });
      el("text", {
        x: m.l - 10,
        y: Y(b) + 4,
        "text-anchor": "end",
      }).textContent = `${b}`;
    }
    for (const z of [10, 15, 20, 25, 30]) {
      el("text", {
        x: X(z),
        y: H - m.b + 22,
        "text-anchor": "middle",
      }).textContent = `${z}`;
    }
    el("text", { x: m.l - 10, y: m.t - 6, "text-anchor": "end" }).textContent =
      "mT";
    el("text", {
      x: (m.l + W - m.r) / 2,
      y: H - 6,
      "text-anchor": "middle",
    }).textContent = "height above the plate's centre (mm)";
    const defs = el("defs", {});
    const curves = [
      ["magnet alone", "#7f8b9d", "2", "2 5", "the magnet alone"],
      ["finite elements", C.fem, "6", "", "finite elements"],
      ["face charge", C.fc, "2.5", "9 6", "face charge"],
      ["cells", C.cells, "2.5", "1.5 5", "cells"],
    ];
    const clips = curves.map(([key, stroke, width, dash, name], i) => {
      const clip = el("clipPath", { id: `mrl-clip-${i}` }, defs);
      const r = el("rect", { x: 0, y: 0, width: 0, height: H }, clip);
      const d = zs
        .map(
          (z, j) =>
            `${j ? "L" : "M"}${X(z).toFixed(1)},${Y(a[key][j]).toFixed(1)}`,
        )
        .join("");
      el("path", {
        d,
        fill: "none",
        stroke,
        "stroke-width": width,
        "stroke-dasharray": dash,
        "stroke-linecap": "round",
        "stroke-opacity": key === "finite elements" ? 0.35 : 1,
        "clip-path": `url(#mrl-clip-${i})`,
      });
      // the solvers' names stacked at the end of their one line, the
      // magnet's under its own
      const end = a[key].at(-1);
      const t = el("text", {
        x: W - m.r + 12,
        y: Y(end) + 5 + (key === "magnet alone" ? 0 : 18 * (i - 2)),
        class: "mrl-label",
        fill: key === "finite elements" ? C.fem : stroke,
        opacity: 0,
      });
      t.textContent = name;
      return { r, t };
    });
    const sect = root.querySelector(".mrl-agree");
    const windows = [
      [0.0, 0.25],
      [0.15, 0.45],
      [0.35, 0.65],
      [0.55, 0.85],
    ];
    return () => {
      const r = sect.getBoundingClientRect();
      const p = reduced
        ? 1
        : clamp((window.innerHeight * 0.95 - r.top) / (r.height * 0.75));
      clips.forEach(({ r: box, t }, i) => {
        const k = smooth(windows[i][0], windows[i][1], p);
        box.setAttribute("width", String(m.l + k * (W - m.l)));
        t.setAttribute(
          "opacity",
          String(smooth(windows[i][1] - 0.05, windows[i][1] + 0.05, p)),
        );
      });
    };
  }

  function seen(selector) {
    const el = root.querySelector(selector);
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          el.classList.add("seen");
          if (el.classList.contains("mrl-code")) {
            el.querySelectorAll(".mrl-line").forEach((line, i) => {
              line.style.transitionDelay = `${i * 0.09}s`;
            });
          }
          io.disconnect();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
  }

  function fill() {
    const sv = data.solvers;
    const a = data.axis;
    const ref = a["finite elements"];
    const err = (key) =>
      Math.max(
        ...a[key].map((b, i) => Math.abs(b - ref[i]) / Math.abs(ref[i])),
      ) * 100;
    const secs = (x) => `${fmt(x, x < 1 ? 2 : 1)} s`;
    const values = {
      gain: `+${fmt(data.gain)}`,
      "bz-plate": `${fmt(data.chi.at(-1).bz)} mT`,
      "bz-alone": `${fmt(data.chi[0].bz)} mT`,
      "force-contact": `${fmt(data.lift[0].force)} N`,
      "force-far": `${fmt(data.lift.at(-1).force, 1)} N`,
      panels: `${fmt(sv.panels)} panels`,
      cells: `${fmt(sv["cell count"])} cells`,
      elements: `${fmt(sv["fem elements"])} elements`,
      "err-face charge": `${fmt(err("face charge"), 2)} %`,
      "err-cells": `${fmt(err("cells"), 1)} %`,
      machine: data.machine ? `an ${data.machine}` : "one laptop",
    };
    for (const key of ["face charge", "cells", "finite elements"])
      values[`t-${key}`] = secs(sv.seconds[key]);
    for (const el of root.querySelectorAll("[data-fill]")) {
      const v = values[el.dataset.fill];
      if (v !== undefined) el.textContent = v;
    }
  }

  // ── the loop ────────────────────────────────────────────────────────────

  let visible = true;
  let pending = false;
  let updateChart = () => {};

  function frame(time) {
    pending = false;
    now = reduced ? 0 : time;
    if (intro !== null && now - intro > 2400) intro = null;
    T = clock();
    words();
    draw();
    updateChart();
    if (visible && !reduced) request();
  }

  function request() {
    if (!pending) {
      pending = true;
      requestAnimationFrame(frame);
    }
  }

  new IntersectionObserver((entries) => {
    visible = entries.some((e) => e.isIntersecting);
    if (visible) request();
  }).observe(root.querySelector(".mrl-story"));
  window.addEventListener("scroll", request, { passive: true });
  window.addEventListener("resize", resize);

  root.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", () => {
      navigator.clipboard?.writeText(button.dataset.copy).then(() => {
        const tag = button.querySelector("span");
        tag.textContent = "copied";
        setTimeout(() => (tag.textContent = "copy"), 1500);
      });
    });
  });

  load().then((d) => {
    data = d;
    data.chiLines = d.chi.map((f) => ({ lines: decode(f) }));
    data.liftLines = d.lift.map((f) => ({ lines: decode(f) }));
    const sigma = d.solvers["face charge"]
      .flatMap((s) => [Math.abs(s[4]), Math.abs(s[5])])
      .sort((x, y) => x - y);
    data.sigmaLevel = sigma[Math.floor(0.7 * (sigma.length - 1))];
    data.cellsMax = Math.max(
      ...d.solvers.cells.map((c) => Math.hypot(c[4], c[5])),
    );
    data.gain = (100 * (d.chi.at(-1).bz - d.chi[0].bz)) / d.chi[0].bz;
    fill();
    updateChart = chart();
    seen(".mrl-agree");
    seen(".mrl-code");
    resize();
    if (!reduced && clock() < 0.3) intro = performance.now();
    request();
  });
  resize();
})();
