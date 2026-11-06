---
html_theme.sidebar_secondary.remove: true
---

# The solver map

Each solver, box by box. Pick a model to light up its path through the map, and
click a box for what happens inside it and the papers it rests on.

```{raw} html
<iframe id="solver-map" src="../_static/map/index.html" title="Map of the three solvers"
  style="width: 100%; height: 2400px; border: 0;" scrolling="no"></iframe>
<script>
  // one scrollbar: the frame takes the height of the map page, also after a
  // model is picked or the window is resized
  (() => {
    const frame = document.getElementById("solver-map");
    const fit = () => {
      const doc = frame.contentDocument;
      if (doc && doc.body) frame.style.height = doc.documentElement.scrollHeight + "px";
    };
    frame.addEventListener("load", () => {
      fit();
      new ResizeObserver(fit).observe(frame.contentDocument.body);
    });
    window.addEventListener("resize", fit);
  })();
</script>
```
