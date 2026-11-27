import {useLayoutEffect, useRef} from "react";
import {format} from "./shared";

const seriesStyles = {
  standard: ["--chart-primary", []],
  nav: ["--chart-primary", []],
  stress: ["--chart-secondary", [8, 3]],
  benchmark: ["--chart-reference", [3, 4]],
};

export default function Chart({points, keys, selected, backtest = false, onSelect, label, id}) {
  const canvas = useRef(null);
  const plot = useRef(null);
  const redraw = useRef(null);

  useLayoutEffect(() => {
    const element = canvas.current;
    redraw.current = () => {
      const width = element.clientWidth;
      const height = element.clientHeight;
      if (!width || !height || !points.length) return;
      const ratio = window.devicePixelRatio || 1;
      const context = element.getContext("2d");
      element.width = Math.round(width * ratio);
      element.height = Math.round(height * ratio);
      context.scale(ratio, ratio);

      const style = getComputedStyle(element);
      const probe = element.firstElementChild;
      const palette = new Map();
      function color(token) {
        if (!palette.has(token)) {
          probe.style.color = "var(" + token + ")";
          palette.set(token, getComputedStyle(probe).color);
        }
        return palette.get(token);
      }
      const left = backtest ? 55 : 43;
      const top = 15;
      const right = width - 12;
      const bottom = height - 28;
      const last = points.length - 1;
      const x = (index) => left + (backtest ? (index + 1) / points.length : index / Math.max(1, last)) * (right - left);
      let min = backtest ? 0.995 : Infinity;
      let max = backtest ? 1.005 : -Infinity;
      for (const point of points) {
        for (const key of keys) {
          min = Math.min(min, point[key]);
          max = Math.max(max, point[key]);
        }
      }
      if (!backtest) {
        min = Math.floor(min * 20) / 20;
        max = Math.ceil(max * 20) / 20;
        if (min === max) { min -= 0.05; max += 0.05; }
      }
      const y = (value) => bottom - (value - min) / (max - min) * (bottom - top);
      plot.current = {left, width: right - left};

      context.font = style.font;
      context.fillStyle = color("--muted");
      context.strokeStyle = color("--line");
      context.textAlign = "right";
      for (let tick = 0; tick <= 4; tick++) {
        const value = min + (max - min) * tick / 4;
        const ordinate = y(value);
        context.beginPath();
        context.moveTo(left, ordinate);
        context.lineTo(right, ordinate);
        context.stroke();
        context.fillText(format(value, backtest ? 3 : 2), left - 6, ordinate + 4);
      }
      const tickCount = Math.min(points.length, backtest ? 2 : width < 400 ? 3 : 5);
      for (let tick = 0; tick < tickCount; tick++) {
        const index = Math.round(last * tick / Math.max(1, tickCount - 1));
        context.textAlign = tick === 0 ? "left" : tick === tickCount - 1 ? "right" : "center";
        const abscissa = tick === tickCount - 1 && tick > 0 ? right : x(index);
        context.fillText(backtest ? points[index].date : points[index].date.slice(0, 7), abscissa, height - 8);
      }

      for (const key of [...keys].reverse()) {
        const [token, dash] = seriesStyles[key];
        context.strokeStyle = color(token);
        context.lineWidth = key === "standard" || key === "nav" ? 2 : 1.5;
        context.setLineDash(dash);
        context.beginPath();
        if (backtest) context.moveTo(left, y(1));
        points.forEach((point, index) => context.lineTo(x(index), y(point[key])));
        context.stroke();
      }

      if (selected == null) return;
      const index = Math.min(selected, last);
      if (!backtest) {
        context.strokeStyle = color("--chart-reference");
        context.lineWidth = 1;
        context.setLineDash([3, 3]);
        context.beginPath();
        context.moveTo(x(index), top);
        context.lineTo(x(index), bottom);
        context.stroke();
      }
      for (const key of keys) {
        context.fillStyle = color(seriesStyles[key][0]);
        context.beginPath();
        context.arc(x(index), y(points[index][key]), 3, 0, 2 * Math.PI);
        context.fill();
      }
    };
    redraw.current();
  }, [points, keys, selected, backtest]);

  useLayoutEffect(() => {
    const draw = () => redraw.current();
    const scheme = matchMedia("(prefers-color-scheme: dark)");
    const observer = new ResizeObserver(draw);
    observer.observe(canvas.current);
    const rootTheme = new MutationObserver(draw);
    rootTheme.observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
    scheme.addEventListener("change", draw);
    return () => {
      observer.disconnect();
      rootTheme.disconnect();
      scheme.removeEventListener("change", draw);
    };
  }, []);

  function move(event) {
    if (!onSelect || !plot.current) return;
    const {left, width} = plot.current;
    const offset = event.clientX - event.currentTarget.getBoundingClientRect().left;
    onSelect(Math.max(0, Math.min(points.length - 1, Math.round((offset - left) / width * (points.length - 1)))));
  }

  function keyDown(event) {
    if (!onSelect || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const last = points.length - 1;
    const step = event.key === "ArrowLeft" ? -1 : 1;
    onSelect(Math.max(0, Math.min(last, (selected ?? last) + step)));
  }

  return (
    <canvas ref={canvas} id={id} role="img" aria-label={label} tabIndex={onSelect ? 0 : undefined}
      onPointerMove={move} onPointerLeave={() => onSelect?.(null)} onKeyDown={keyDown}
      onBlur={() => onSelect?.(null)}><span /></canvas>
  );
}
