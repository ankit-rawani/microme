"""Web chat UI for the tiny LLM (plan §10). Serves micro_125m_mix directly
(PyTorch, no GGUF) with a live RAG toggle so you can see grounded vs raw answers.

  python serve.py     # -> http://localhost:8000
"""
import rag  # loads model + embedder + facts once
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>MicroMe</title>
<style>
:root{color-scheme:dark}
body{margin:0;font:15px/1.5 system-ui,sans-serif;background:#0e1116;color:#e6e6e6;display:flex;flex-direction:column;height:100vh}
header{padding:12px 18px;border-bottom:1px solid #222;display:flex;align-items:center;gap:14px}
header b{font-size:16px} header .sub{color:#888;font-size:13px}
label.tog{margin-left:auto;font-size:13px;color:#bbb;cursor:pointer;user-select:none}
#log{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:760px;padding:10px 14px;border-radius:12px;white-space:pre-wrap}
.u{align-self:flex-end;background:#2a4d69}
.a{align-self:flex-start;background:#1b212b}
.facts{align-self:flex-start;max-width:760px;font-size:12px;color:#7fa;opacity:.8;padding:0 6px}
form{display:flex;gap:8px;padding:14px 18px;border-top:1px solid #222}
input{flex:1;padding:11px 14px;border-radius:10px;border:1px solid #333;background:#12161d;color:#eee;font-size:15px}
button{padding:11px 20px;border:0;border-radius:10px;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}
</style></head><body>
<header><b>MicroMe</b><span class="sub">125M · trained from scratch on an RTX 4060</span>
<label class="tog"><input type="checkbox" id="rag" checked> RAG (grounded)</label></header>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Ask something…" autocomplete="off" autofocus>
<button id="b">Send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),b=document.getElementById('b');
function add(cls,txt){const d=document.createElement('div');d.className='msg '+cls;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const m=q.value.trim();if(!m)return;
 add('u',m);q.value='';b.disabled=true;const wait=add('a','…');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({message:m,use_rag:document.getElementById('rag').checked})});
  const j=await r.json();wait.textContent=j.reply||'(no reply)';
  if(j.facts&&j.facts.length){const fd=document.createElement('div');fd.className='facts';
   fd.textContent='📎 retrieved: '+j.facts.map(f=>f.slice(0,70)+'…').join('  |  ');log.appendChild(fd);}
 }catch(err){wait.textContent='error: '+err;}
 b.disabled=false;q.focus();log.scrollTop=log.scrollHeight;};
</script></body></html>"""


class Msg(BaseModel):
    message: str
    use_rag: bool = True


@app.get("/")
def index():
    return HTMLResponse(PAGE)


@app.post("/chat")
def chat(m: Msg):
    if m.use_rag:
        reply, facts = rag.answer(m.message)
    else:
        reply, facts = rag.gen(m.message), []
    return {"reply": reply, "facts": list(facts)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
