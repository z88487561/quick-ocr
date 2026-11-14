// PaddleOCR（PP-OCRv5）背景執行緒：載入 onnxruntime-web 與模型，接收影像、回傳文字行。
// 選了 WebGPU 時，辨識／方向分類走 GPU（ort.jspi 核心，需 WebAssembly JSPI：Chrome／Edge 137 以上）；偵測固定在 CPU。
// 不可用、自檢失敗或執行出錯時自動退回 WASM（CPU）。
import {PPOCR} from './pp-ocr.mjs';
const base=new URL('./',import.meta.url);
const FILES=[['PP-OCRv5_mobile_det_infer.ort',4896928],['PP-OCRv5_mobile_rec_infer.onnx',16559278],['cls.onnx',585000],['ppocrv5_dict.txt',74014]];
let eng=null,initP=null,got=0,backend='wasm',accel='cpu',threadPref='auto',nThreads=1,models=null,gpuNote={asked:false};
const total=FILES.reduce((a,f)=>a+f[1],0);
const post=m=>self.postMessage(m);
async function fetchBytes(name){
  const r=await fetch(new URL(name,base));
  if(!r.ok)throw new Error(name+' HTTP '+r.status);
  const rd=r.body.getReader(),chunks=[];let n=0;
  for(;;){const {done,value}=await rd.read();if(done)break;chunks.push(value);n+=value.length;got+=value.length;post({t:'load',f:Math.min(1,got/total)})}
  const out=new Uint8Array(n);let o=0;for(const c of chunks){out.set(c,o);o+=c.length}
  return out;
}
// 探測 WebGPU：回傳 {ok,name} 或 {ok:false,reason}（reason 會顯示給使用者，方便判斷為何沒用上 GPU）
async function gpuProbe(){
  if(!(self.navigator&&navigator.gpu))return {ok:false,reason:'Worker 內沒有 navigator.gpu'};
  if(!('Suspending' in WebAssembly))return {ok:false,reason:'此瀏覽器不支援 WebAssembly JSPI（WebGPU 選項需要 Chrome／Edge 137 以上）'};
  let a=null,err='';
  for(const o of [{powerPreference:'high-performance'},{}]){
    try{a=await navigator.gpu.requestAdapter(o);if(a)break}catch(e){err=String(e&&e.message||e)}
  }
  if(!a)return {ok:false,reason:'找不到可用的 GPU 介面卡（requestAdapter 回傳空值'+(err?'：'+err.slice(0,80):'')+'），請在 chrome://gpu 查看 WebGPU 狀態'};
  let info={};try{info=a.info||(a.requestAdapterInfo&&await a.requestAdapterInfo())||{}}catch{}
  return {ok:true,name:[info.vendor,info.architecture||info.description].filter(Boolean).join(' ')};
}
const short=e=>String(e&&e.message||e).replace(/\s+/g,' ').slice(0,120);
// 建立引擎：name = 'webgpu' | 'wasm'
async function build(name){
  const ort=await import(name==='webgpu'?'./ort.jspi.min.mjs':'./ort.wasm.min.mjs');
  ort.env.wasm.wasmPaths=base.href;
  ort.env.wasm.proxy=false;
  ort.env.wasm.numThreads=nThreads;
  const cpu={executionProviders:['wasm'],graphOptimizationLevel:'all'};
  // 偵測模型為 .ort 格式（已針對 CPU 最佳化），WebGPU 無法載入，固定走 CPU；辨識／方向分類走 GPU
  const opt=name==='webgpu'?{det:cpu,rec:{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'},cls:{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'}}:cpu;
  return PPOCR.create(ort,{det:models.det,rec:models.rec,cls:models.cls,dict:models.dict},opt);
}
// GPU 自檢：畫一行英數字，確認偵測＋辨識的結果正常（避免 GPU 驅動錯誤造成「不報錯但全是亂碼」）
async function canary(e){
  if(typeof OffscreenCanvas==='undefined')return;
  const W=420,H=84,c=new OffscreenCanvas(W,H),g=c.getContext('2d');
  g.fillStyle='#fff';g.fillRect(0,0,W,H);g.fillStyle='#000';g.font='bold 44px sans-serif';g.textBaseline='middle';g.fillText('OCR 2026',18,44);
  const d=g.getImageData(0,0,W,H);
  const r=await e.recognize({data:d.data,width:W,height:H},{detScales:[960]},()=>{});
  const t=r.map(l=>l.text).join('').replace(/\s/g,'').toUpperCase();
  if(!/2026|OCR/.test(t))throw new Error('WebGPU 自檢結果不符：'+JSON.stringify(t));
}
async function useWasm(reason){
  eng=await build('wasm');backend='wasm';
  gpuNote={asked:true,ok:false,reason};
  post({t:'fallback',reason});
}
// 執行緒數：需要 cross-origin isolation（SharedArrayBuffer）才能多執行緒。
// 自動＝邏輯核心數的一半（避開超執行緒），最多 4；也可手動指定
function pickThreads(){
  if(!self.crossOriginIsolated||typeof SharedArrayBuffer==='undefined')return 1;
  if(threadPref!=='auto'){const n=parseInt(threadPref,10);return Math.max(1,Math.min(Number.isFinite(n)?n:1,16))}
  const hc=(self.navigator&&navigator.hardwareConcurrency)||2;
  return Math.max(1,Math.min(4,Math.floor(hc/2)));
}
function init(){
  if(initP)return initP;
  initP=(async()=>{
    nThreads=pickThreads();
    const [det,rec,cls,dict]=await Promise.all(FILES.map(f=>fetchBytes(f[0])));
    models={det,rec,cls,dict:new TextDecoder().decode(dict)};
    gpuNote={asked:accel!=='cpu'};
    const pr=gpuNote.asked?await gpuProbe():{ok:false};
    if(gpuNote.asked&&!pr.ok)gpuNote={asked:true,ok:false,reason:pr.reason};
    post({t:'stage',text:pr.ok?'載入運算引擎（WebGPU，首次約 18MB）…':'載入運算引擎（CPU，首次約 15MB）…',f:1});
    if(pr.ok){
      let built=false;
      try{eng=await build('webgpu');built=true}
      catch(err){console.warn('WebGPU 載入失敗：',err);await useWasm('GPU 載入模型失敗：'+short(err))}
      if(built){
        try{post({t:'stage',text:'GPU 自檢中…',f:1});await canary(eng);backend='webgpu';gpuNote={asked:true,ok:true,name:pr.name}}
        catch(err){console.warn('WebGPU 自檢失敗：',err);await useWasm('GPU 自檢結果不符：'+short(err))}
      }
    }else{eng=await build('wasm');backend='wasm'}
    post({t:'ready',backend,gpu:gpuNote,threads:nThreads,isolated:!!self.crossOriginIsolated});
  })();
  initP.catch(e=>{initP=null;got=0;post({t:'initfail',message:String(e&&e.message||e)})});
  return initP;
}
self.onmessage=async e=>{
  const m=e.data;
  try{
    if(m.t==='init'){accel=m.accel||'cpu';threadPref=String(m.threads||'auto');await init()}
    else if(m.t==='run'){
      await init();
      const img={data:new Uint8ClampedArray(m.rgba),width:m.w,height:m.h};
      const go=()=>eng.recognize(img,m.opts||{},(stage,f)=>post({t:'prog',id:m.id,stage,f}));
      let lines;
      try{lines=await go()}
      catch(err){
        if(backend!=='webgpu')throw err;
        console.warn('WebGPU 執行失敗，改用 CPU 重試：',err);
        await useWasm('GPU 執行失敗：'+short(err));lines=await go();
      }
      post({t:'result',id:m.id,lines,backend,gpu:gpuNote,threads:nThreads});
    }
  }catch(err){post({t:'error',id:m.id,message:String(err&&err.message||err)})}
};
