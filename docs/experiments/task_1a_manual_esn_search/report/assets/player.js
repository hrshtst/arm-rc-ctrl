/* Copyright (c) 2026 Hiroshi Atsuta · SPDX-License-Identifier: GPL-3.0-only */
/* Presentation only: hold the nearest preceding retained sample; never extrapolate a run. */
(() => {
  "use strict";
  function indexAt(times, t) {
    let lo = 0, hi = times.length - 1;
    while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (times[mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
  }
  function points(q, lengths) {
    if (!q || q.some(v => v === null || !Number.isFinite(v))) return null;
    let x = 0, y = 0, angle = 0; const result = [[0, 0]];
    q.forEach((v, i) => { angle += v; x += lengths[i] * Math.cos(angle); y += lengths[i] * Math.sin(angle); result.push([x, y]); });
    return result;
  }
  let activePause = () => {};
  for (const data of window.manualCases) {
    const player = document.querySelector(`[data-case="${data.slug}"]`);
    if (!player) continue;
    const slider = player.querySelector(".timeline"), clock = player.querySelector(".clock");
    const button = player.querySelector(".play"), speed = player.querySelector(".speed");
    slider.min = String(-data.activation); slider.value = slider.min;
    const panels = data.runs.map(run => {
      const panel = document.createElement("div"); panel.className = "anim-panel"; panel.style.setProperty("--role", run.color);
      const title = document.createElement("h4"); title.textContent = run.role;
      const verdict = document.createElement("span"); verdict.className = run.success ? "pass" : "fail"; verdict.textContent = run.success ? "PASS" : "FAIL"; title.append(verdict);
      const canvas = document.createElement("canvas"); canvas.width = 380; canvas.height = 400;
      canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", `${data.title}: ${run.role} recorded arm motion`);
      const status = document.createElement("div"); status.className = "anim-status";
      panel.append(title, canvas, status); player.querySelector(".canvases").append(panel);
      return {run, canvas, status, actual:run.q.map(q => points(q,data.lengths)), reference:run.ref.map(q => points(q,data.lengths))};
    });
    function draw(now) {
      clock.value = `${now.toFixed(2)} s`;
      for (const panel of panels) {
        const {run, canvas, status} = panel, ctx = canvas.getContext("2d");
        const i = indexAt(run.t, now), ended = now > run.end + 0.005;
        ctx.clearRect(0,0,380,400);
        const xy = ([x,y]) => [190+x*305,195-y*305];
        ctx.strokeStyle="#e7edf0";ctx.lineWidth=1;
        for (const v of [-0.5,-0.25,0,0.25,0.5]) {
          const [a,b]=xy([v,-0.55]),[c,d]=xy([v,0.55]);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke();
          const [e,f]=xy([-0.55,v]),[g,h]=xy([0.55,v]);ctx.beginPath();ctx.moveTo(e,f);ctx.lineTo(g,h);ctx.stroke();
        }
        ctx.fillStyle="#617682";ctx.font="16px system-ui";ctx.fillText("−0.5",23,382);ctx.fillText("0",185,382);ctx.fillText("0.5 m",318,382);
        const [tx,ty]=xy(data.target);ctx.strokeStyle="#172c39";ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(tx,ty,data.radius*305,0,2*Math.PI);ctx.stroke();
        function line(pts,color,width,dashed=false) {
          if(!pts)return;ctx.beginPath();ctx.setLineDash(dashed?[6,5]:[]);ctx.strokeStyle=color;ctx.lineWidth=width;
          pts.forEach((p,j)=>{const [x,y]=xy(p);if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.setLineDash([]);
        }
        line(panel.actual.slice(0,i+1).filter(Boolean).map(p=>p[p.length-1]),ended?"#b9bfc2":run.color,1.5);
        line(panel.reference[i],ended?"#c9cdd0":run.color+"88",2.5,true);
        line(panel.actual[i],ended?"#a1a9ae":run.color,5);
        if(panel.actual[i])for(const p of panel.actual[i]){const [x,y]=xy(p);ctx.fillStyle=ended?"#a1a9ae":run.color;ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fill();}
        const pulse = run.pulse.length===2 && now>=run.pulse[0] && now<=run.pulse[1];
        status.textContent = ended ? `Record ended at ${run.end.toFixed(2)} s; last state only` : (now<0?"Warm-up":pulse?"FORCE PULSE ACTIVE":"Active motion / hold");
        if(pulse){ctx.fillStyle="#a84322";ctx.font="bold 16px system-ui";ctx.fillText("12 N pulse",12,25);}
        if(!ended)status.textContent += ` · sample ${run.t[i].toFixed(2)} s`;
      }
    }
    let playing=false,last=0,handle=null;
    function pause(){playing=false;button.textContent="Play";if(handle!==null)cancelAnimationFrame(handle);handle=null;}
    function tick(timestamp){if(!playing)return;const dt=Math.min((timestamp-last)/1000,0.25);last=timestamp;
      const t=Math.min(30,Number(slider.value)+dt*Number(speed.value));slider.value=String(t);draw(t);
      if(t>=30)pause();else handle=requestAnimationFrame(tick);
    }
    button.addEventListener("click",()=>{if(playing){pause();return;}activePause();activePause=pause;
      if(Number(slider.value)>=30)slider.value=slider.min;playing=true;button.textContent="Pause";last=performance.now();handle=requestAnimationFrame(tick);
    });
    player.querySelector(".reset").addEventListener("click",()=>{pause();slider.value=slider.min;draw(Number(slider.value));});
    slider.addEventListener("input",()=>{pause();draw(Number(slider.value));});
    document.addEventListener("visibilitychange",()=>{if(document.hidden)pause();});
    draw(Number(slider.value));
  }
})();
