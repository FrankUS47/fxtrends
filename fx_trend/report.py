"""產生靜態網頁 docs/index.html（Plotly K 線 + 分析面板），供 GitHub Pages 發佈。"""
from __future__ import annotations

import json
import math

from . import config as C

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>專業匯率 K 線分析與買賣預測系統</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="https://s3.tradingview.com/tv.js"></script>
<style>
:root{--bg:#121721;--bg2:#1e2433;--card:#283044;--tx:#d1d5db;--mut:#9ca3af;--g:#26a69a;--r:#ef5350;--b:#2962ff;--y:#ffd700;--bd:#374151}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif;background:var(--bg);color:var(--tx);padding:16px;line-height:1.5}
.container{max-width:1500px;margin:0 auto;display:grid;grid-template-columns:1fr 380px;gap:16px}
header{grid-column:1/-1;background:var(--bg2);padding:16px 20px;border-radius:8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;border:1px solid var(--bd)}
h1{font-size:1.3rem;color:#fff}.sub{font-size:.82rem;color:var(--mut)}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap}label{font-size:.85rem;color:var(--mut)}
select{background:var(--card);color:var(--tx);border:1px solid var(--bd);padding:8px 12px;border-radius:6px;font-size:.9rem}
main,aside{display:flex;flex-direction:column;gap:16px;min-width:0}
.box{background:var(--bg2);border-radius:8px;border:1px solid var(--bd);padding:14px}
.box h3{font-size:1rem;color:#fff;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--bd)}
#chart{height:540px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.card{background:var(--card);border-radius:6px;padding:12px;border:1px solid var(--bd)}
.card h4{font-size:.95rem;color:var(--y);margin-bottom:8px}.card p,.card li{font-size:.85rem;line-height:1.6}
.card ul{list-style:none}.card li::before{content:"▪ ";color:var(--b)}
.score{text-align:center;padding:8px 0}.score .v{font-size:2.4rem;font-weight:700}.score .l{font-size:.85rem;font-weight:700}
.act{margin-top:8px;padding:10px;border-radius:6px;text-align:center;font-weight:700;font-size:1.1rem}
.BUY{background:rgba(38,166,154,.2);color:var(--g);border:1px solid var(--g)}
.SELL{background:rgba(239,83,80,.2);color:var(--r);border:1px solid var(--r)}
.HOLD{background:rgba(255,215,0,.15);color:var(--y);border:1px solid var(--y)}
.bull{color:var(--g)}.bear{color:var(--r)}.neutral{color:var(--y)}
.rows{list-style:none;font-size:.82rem}.rows li{padding:5px 0;border-bottom:1px dashed var(--bd);display:flex;justify-content:space-between;gap:8px}
.small{font-size:.75rem;color:var(--mut);margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:.78rem}td,th{padding:4px 6px;border-bottom:1px dashed var(--bd);text-align:right}th{color:var(--mut);font-weight:400}td:first-child,th:first-child{text-align:left}
@media(max-width:1024px){.container{grid-template-columns:1fr}}
</style></head><body>
<div class="container">
<header>
 <div><h1>專業匯率 K 線分析與買賣預測系統</h1>
 <div class="sub">分析更新：__GENERATED__（台北時間，約每 5 分鐘一次）｜TradingView 圖表與報價為即時｜每次更新自動重新訓練並檢討預測</div></div>
 <div class="ctl">
  <div><label>貨幣對：</label><select id="pair"></select></div>
  <div><label>圖表：</label><select id="view"><option value="TV">TradingView 即時圖</option><option value="PY">Python K線（含買賣訊號）</option></select></div>
  <div><label>週期：</label><select id="tf"><option value="D">日線</option><option value="W">週線</option><option value="M">月線</option></select></div>
 </div>
</header>
<main>
 <div class="box" style="padding:8px;min-height:120px">
  <div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-forex-cross-rates.js" async>
  {"width":"100%","height":"115","currencies":["USD","EUR","JPY","GBP","TWD","CHF"],"isTransparent":true,"colorTheme":"dark","locale":"zh_TW"}
  </script></div>
 </div>
 <div class="box"><div id="chart"></div><div id="tvchart" style="height:540px;display:none"></div>
  <div class="small" id="chartnote">▲ 買進 / ▼ 賣出標記為「樣本外 walk-forward」歷史訊號（當時只用之前的資料訓練），非事後諸葛。</div></div>
 <div class="box"><div class="cards">
  <div class="card"><h4>📊 支撐與壓力位</h4><ul id="sr"></ul></div>
  <div class="card"><h4>🔄 反轉與續勢型態</h4><div id="pat"></div></div>
  <div class="card"><h4>📈 趨勢線與區間預測</h4><p id="trend"></p></div>
 </div></div>
</main>
<aside>
 <div class="box" id="quote" style="padding:4px;min-height:100px"></div>
 <div class="box"><h3>買賣訊號</h3>
  <div class="score"><div class="v" id="score">--</div><div class="l" id="lab"></div></div>
  <div class="act" id="act"></div>
  <ul class="rows" id="factors"></ul>
  <div class="small" id="actnote"></div></div>
 <div class="box"><h3>模型可信度（請務必參考）</h3><table id="stats"></table><div class="small" id="statnote"></div></div>
 <div class="box"><h3>近期預測紀錄與對答案</h3><table id="log"></table></div>
 <div class="box"><div class="small">本頁為量化研究工具，並非投資建議。外匯接近有效市場，樣本外命中率常只略高於 50%；請以上方「模型可信度」為準，並自行承擔交易風險。</div></div>
</aside>
</div>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);
const pairSel=$('pair'),tfSel=$('tf'),viewSel=$('view');
Object.entries(DATA.pairs).forEach(([k,v])=>pairSel.add(new Option(v.label,k)));
try{const st=JSON.parse(localStorage.getItem('fx_state')||'{}');
 if(st.pair&&DATA.pairs[st.pair])pairSel.value=st.pair;if(st.tf)tfSel.value=st.tf;if(st.view)viewSel.value=st.view;}catch(e){}
const save=()=>{try{localStorage.setItem('fx_state',JSON.stringify({pair:pairSel.value,tf:tfSel.value,view:viewSel.value}))}catch(e){}};
let tvKey='',qKey='';
function mountQuote(P){
 if(qKey===P.tv)return;qKey=P.tv;const q=$('quote');q.innerHTML='';
 const w=document.createElement('div');w.className='tradingview-widget-container';
 const s=document.createElement('script');s.src='https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js';s.async=true;
 s.text=JSON.stringify({symbol:P.tv,width:'100%',colorTheme:'dark',isTransparent:true,locale:'zh_TW'});
 w.appendChild(s);q.appendChild(w);}
function applyView(P,tf){
 const tv=viewSel.value==='TV';
 $('chart').style.display=tv?'none':'block';$('tvchart').style.display=tv?'block':'none';$('chartnote').style.display=tv?'none':'block';
 if(tv){const k=P.tv+tf;
  if(k!==tvKey&&typeof TradingView!=='undefined'){tvKey=k;$('tvchart').innerHTML='';
   new TradingView.widget({autosize:true,symbol:P.tv,interval:tf,timezone:'Asia/Taipei',theme:'dark',style:'1',locale:'zh_TW',enable_publishing:false,allow_symbol_change:false,container_id:'tvchart'});}}
 mountQuote(P);}
const pct=(x,d=1)=>x==null?'—':(x*100).toFixed(d)+'%';
function render(){
 const P=DATA.pairs[pairSel.value],tf=tfSel.value,c=P.candles[tf],dp=P.decimals,f=x=>x.toFixed(dp);
 $('chart').style.display='block';
 const tr=[{type:'candlestick',x:c.x,open:c.o,high:c.h,low:c.l,close:c.c,name:'K線',increasing:{line:{color:'#26a69a'}},decreasing:{line:{color:'#ef5350'}}},
  {x:c.x,y:c.s20,name:'MA20',mode:'lines',line:{color:'#ffd700',width:1}},
  {x:c.x,y:c.s60,name:'MA60',mode:'lines',line:{color:'#2962ff',width:1}},
  {x:c.x,y:c.bu,name:'布林上',mode:'lines',line:{color:'#6b7280',width:1,dash:'dot'}},
  {x:c.x,y:c.bl,name:'布林下',mode:'lines',line:{color:'#6b7280',width:1,dash:'dot'}}];
 if(tf==='D'){
  tr.push({x:P.markers.buy.map(m=>m.x),y:P.markers.buy.map(m=>m.y),mode:'markers',name:'買進訊號',marker:{symbol:'triangle-up',size:9,color:'#26a69a'}});
  tr.push({x:P.markers.sell.map(m=>m.x),y:P.markers.sell.map(m=>m.y),mode:'markers',name:'賣出訊號',marker:{symbol:'triangle-down',size:9,color:'#ef5350'}});}
 const L=P.levels,line=(y,col,t)=>({type:'line',xref:'paper',x0:0,x1:1,y0:y,y1:y,line:{color:col,width:1,dash:'dash'}}),
  ann=(y,col,t)=>({xref:'paper',x:1,y:y,text:t,showarrow:false,xanchor:'right',yanchor:'bottom',font:{color:col,size:10}});
 Plotly.react('chart',tr,{paper_bgcolor:'#1e2433',plot_bgcolor:'#1e2433',font:{color:'#d1d5db'},margin:{l:50,r:50,t:10,b:30},
  xaxis:{rangeslider:{visible:false},type:'category',nticks:10,gridcolor:'#2f3648'},yaxis:{side:'right',gridcolor:'#2f3648'},legend:{orientation:'h',y:1.06},
  shapes:[line(L.resistance,'#ef5350'),line(L.support,'#26a69a'),line(L.pivot.P,'#9ca3af')],
  annotations:[ann(L.resistance,'#ef5350','壓力 '+f(L.resistance)),ann(L.support,'#26a69a','支撐 '+f(L.support)),ann(L.pivot.P,'#9ca3af','樞軸 '+f(L.pivot.P))]},{responsive:true});
 $('sr').innerHTML=`<li><b>壓力位：</b>${f(L.resistance)}（${L.resistance_touches||'ATR 推估'} 次觸及）</li><li><b>R1 / R2：</b>${f(L.pivot.R1)} / ${f(L.pivot.R2)}</li><li><b>樞軸點：</b>${f(L.pivot.P)}</li><li><b>S1 / S2：</b>${f(L.pivot.S1)} / ${f(L.pivot.S2)}</li><li><b>支撐位：</b>${f(L.support)}（${L.support_touches||'ATR 推估'} 次觸及）</li><li><b>ATR(14)：</b>${f(L.atr)}</li>`;
 $('pat').innerHTML=P.patterns.map(p=>`<p><b class="${p.bias}">【${p.name}】</b> ${p.desc}</p>`).join('');
 const T=P.trend;
 $('trend').innerHTML=`近 60 日趨勢斜率 ${T.slope_pct_per_day>=0?'+':''}${T.slope_pct_per_day.toFixed(3)}%/日（R²=${T.r2.toFixed(2)}），通道 ${f(T.channel_lower)} ~ ${f(T.channel_upper)}。<br>未來 ${T.horizon} 個交易日，以近期波動估算：68% 區間 ${f(T.range68[0])} ~ ${f(T.range68[1])}；95% 區間 ${f(T.range95[0])} ~ ${f(T.range95[1])}。<br><span style="color:var(--mut)">（區間為波動度推估，不含方向判斷）</span>`;
 const cls=P.action==='BUY'?'bull':P.action==='SELL'?'bear':'neutral';
 $('score').className='v '+cls;$('score').textContent=P.score;
 $('lab').className='l '+cls;$('lab').textContent=`未來 ${P.horizon} 日上漲機率 ${(P.proba*100).toFixed(1)}%`;
 $('act').className='act '+P.action;$('act').textContent=P.action_text;
 $('factors').innerHTML=P.factors.map(x=>`<li><span>${x.name}</span><span class="${x.value>=0?'bull':'bear'}">${x.value>=0?'偏多 +':'偏空 '}${x.value.toFixed(2)}</span></li>`).join('');
 $('actnote').textContent=`買進 ≥ ${((0.5+P.margin)*100).toFixed(0)} 分｜賣出 ≤ ${((0.5-P.margin)*100).toFixed(0)} 分｜K 線日期 ${P.bar_date}｜現價 ${f(P.price)}（${P.change_pct>=0?'+':''}${P.change_pct.toFixed(2)}%）`;
 const S=P.stats,row=(n,e)=>e&&e.n?`<tr><td>${n}</td><td>${e.n}</td><td>${pct(e.hit)}</td><td>${pct(e.mean_ret,2)}</td><td>${pct(e.base_up,0)}</td></tr>`:`<tr><td>${n}</td><td>0</td><td>—</td><td>—</td><td>—</td></tr>`;
 const lv=S.live;
 $('stats').innerHTML=`<tr><th>樣本</th><th>訊號數</th><th>方向命中</th><th>每筆淨報酬</th><th>單純上漲率</th></tr>`+row('調參區(樣本外)',S.fit)+row('Holdout(樣本外)',S.holdout)+`<tr><td>實盤(已對答案)</td><td>${lv.n}</td><td>${pct(lv.hit_all)}</td><td>—</td><td>—</td></tr><tr><td>實盤近 ${lv.n_recent} 筆</td><td>${lv.n_recent}</td><td>${pct(lv.hit_recent)}</td><td>—</td><td>—</td></tr>`;
 $('statnote').textContent=`參數：預測 ${P.horizon} 日、L2=${P.params.l2}、門檻 ±${P.margin}｜上次調參 ${P.tuned_at}（${P.tune_reason||''}）。每筆淨報酬已扣成本；Holdout 未參與調參，較可信。`;
 $('log').innerHTML=`<tr><th>K線日</th><th>訊號</th><th>機率</th><th>${'結果'}</th></tr>`+P.log.map(r=>`<tr><td>${r.date}</td><td class="${r.action==='BUY'?'bull':r.action==='SELL'?'bear':'neutral'}">${r.action}</td><td>${(r.p*100).toFixed(0)}%</td><td>${r.fwd==null?'待揭曉('+r.h+'日)':(r.hit==null?'—':(r.hit?'✔ ':'✘ '))+(r.fwd*100).toFixed(2)+'%'}</td></tr>`).join('');
 applyView(P,tf);
}
pairSel.onchange=tfSel.onchange=viewSel.onchange=()=>{save();render();};render();
setTimeout(()=>location.reload(),300000);
</script></body></html>"""


def _clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else round(o, 6)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def build_report(results: dict, generated: str, errors: dict | None = None) -> str:
    payload = {"generated": generated, "pairs": _clean(results), "errors": errors or {}}
    html = TEMPLATE.replace("__GENERATED__", generated)
    return html.replace("__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False))


def write_outputs(results: dict, generated: str, errors: dict | None = None) -> None:
    C.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (C.DOCS_DIR / "index.html").write_text(build_report(results, generated, errors), encoding="utf-8")
    slim = {k: {x: v[x] for x in ("label", "price", "bar_date", "proba", "score", "action", "horizon", "levels", "stats")}
            for k, v in results.items()}
    C.LATEST_FILE.write_text(json.dumps({"generated": generated, "pairs": _clean(slim), "errors": errors or {}},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    (C.DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
