// PaddleOCR（PP-OCRv5）背景執行緒：載入 onnxruntime-web 與模型，接收影像、回傳文字行。
import * as ort from './ort.wasm.min.mjs';
import {PPOCR} from './pp-ocr.mjs';
const base=new URL('./',import.meta.url);
const FILES=[['PP-OCRv5_mobile_det_infer.ort',4896928],['PP-OCRv5_mobile_rec_infer.onnx',16559278],['cls.onnx',585000],['ppocrv5_dict.txt',74014]];
let eng=null,initP=null,got=0;
const total=FILES.reduce((a,f)=>a+f[1],0);
async function fetchBytes(name){
  const r=await fetch(new URL(name,base));
  if(!r.ok)throw new Error(name+' HTTP '+r.status);
  const rd=r.body.getReader(),chunks=[];let n=0;
  for(;;){const {done,value}=await rd.read();if(done)break;chunks.push(value);n+=value.length;got+=value.length;self.postMessage({t:'load',f:Math.min(1,got/total)})}
  const out=new Uint8Array(n);let o=0;for(const c of chunks){out.set(c,o);o+=c.length}
  return out;
}
function init(){
  if(initP)return initP;
  initP=(async()=>{
    ort.env.wasm.wasmPaths=base.href;
    ort.env.wasm.proxy=false;
    // 需要 cross-origin isolation 才能多執行緒；一般頁面為單執行緒
    ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,(self.navigator&&navigator.hardwareConcurrency)||2):1;
    const [det,rec,cls,dict]=await Promise.all(FILES.map(f=>fetchBytes(f[0])));
    eng=await PPOCR.create(ort,{det,rec,cls,dict:new TextDecoder().decode(dict)},{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    self.postMessage({t:'ready'});
  })();
  initP.catch(e=>{initP=null;got=0;self.postMessage({t:'initfail',message:String(e&&e.message||e)})});
  return initP;
}
self.onmessage=async e=>{
  const m=e.data;
  try{
    if(m.t==='init')await init();
    else if(m.t==='run'){
      await init();
      const img={data:new Uint8ClampedArray(m.rgba),width:m.w,height:m.h};
      const lines=await eng.recognize(img,m.opts||{},(stage,f)=>self.postMessage({t:'prog',id:m.id,stage,f}));
      self.postMessage({t:'result',id:m.id,lines});
    }
  }catch(err){self.postMessage({t:'error',id:m.id,message:String(err&&err.message||err)})}
};
