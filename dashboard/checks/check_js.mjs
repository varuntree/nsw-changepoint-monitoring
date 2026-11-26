// Runs the detector code that ships inside index.html (the <script id="engine"> block) under Node and compares it
// with the Python results stored in data.json, and with checks/truth.json for inserted drops.
//   node checks/check_js.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, "..", "index.html"), "utf8");
const src = html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
const Engine = new Function("module", src + "; return Engine;")({});
const data = JSON.parse(fs.readFileSync(path.join(here, "..", "data.json"), "utf8"));
const truthFile = path.join(here, "truth.json");
const truth = fs.existsSync(truthFile) ? JSON.parse(fs.readFileSync(truthFile, "utf8")) : null;
let ok = true;
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i] === null ? NaN : b[i]; if (Number.isNaN(x) !== Number.isNaN(y)) return Infinity; if (!Number.isNaN(x)) m = Math.max(m, Math.abs(x - y)); } return m; };
for (const [key, S] of Object.entries(data.series)) {
  const base = Engine.prepare(S), run = Engine.monitor(base.r, S);
  const jsDates = run.alarms.map(i => Engine.dateAt(S, i).toISOString().slice(0, 10));
  const pyDates = S.alarms.map(a => a.date);
  const same = JSON.stringify(jsDates) === JSON.stringify(pyDates);
  ok &&= same;
  console.log(`\n[${key}] ${S.monitor.label}`);
  console.log(`  alarm dates  JS ${jsDates.length}, Python ${pyDates.length}: ${same ? "IDENTICAL" : "DIFFERENT"}`);
  if (!same) console.log("   JS    ", jsDates.join(" "), "\n   Python", pyDates.join(" "));
  else if (jsDates.length <= 8) console.log("   ", jsDates.join(", "));
  console.log(`  max abs difference  surprise ${maxDiff(base.r, S.r).toExponential(1)}, scale ${maxDiff(run.s, S.s).toExponential(1)}, statistic ${maxDiff(run.stat, S.stat).toExponential(1)}, threshold ${maxDiff(run.h, S.h).toExponential(1)}`);
  if (truth && truth[key]) {
    let exact = 0, within1 = 0, flagAgree = 0, worst = [];
    for (const c of truth[key]) {
      const mod = Engine.insertDrop(S, base, c.pct, c.days), r2 = Engine.monitor(mod.r, S);
      const hit = r2.alarms.find(a => a >= mod.tau), js = hit === undefined ? null : hit - mod.tau + 1;
      if (js === c.after) exact++;
      if ((js === null) === (c.after === null)) { flagAgree++; if (js === null || Math.abs(js - c.after) <= 1) within1++; }
      if (js !== c.after) worst.push(`${c.pct}%/${c.days}d: JS ${js}, Python ${c.after}`);
    }
    const n = truth[key].length;
    console.log(`  inserted drops (${n} cases, full Python pipeline as truth): same day ${exact}/${n}, within 1 day ${within1}/${n}, flagged-or-not agrees ${flagAgree}/${n}`);
    if (worst.length) console.log("   differences:", worst.join("; "));
    if (S.monitor.forecast === "HW") ok &&= exact === n;
  }
}
console.log(ok ? "\nPASS" : "\nFAIL");
process.exit(ok ? 0 : 1);
