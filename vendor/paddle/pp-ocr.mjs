// PP-OCRv5 (det + cls + rec) 純 JS 流程：不依賴 DOM，瀏覽器 Worker 與 Node 皆可執行。
// 輸入：{data: RGBA Uint8(Clamped)Array, width, height}；輸出：每行 {text, conf, quad, chars[]}
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
const MEAN=[0.485,0.456,0.406],STD=[0.229,0.224,0.225];

// ---- 重新取樣（三角濾波、縮小時抗鋸齒）----
function axisWeights(inN,outN){
  const scale=inN/outN,fs=Math.max(scale,1),support=fs,maxk=Math.ceil(2*support)+3;
  const start=new Int32Array(outN),cnt=new Int32Array(outN),wts=new Float32Array(outN*maxk);
  for(let o=0;o<outN;o++){
    const c=(o+.5)*scale;let x0=Math.max(0,Math.floor(c-support)),x1=Math.min(inN,Math.ceil(c+support)),n=x1-x0,sum=0;
    for(let k=0;k<n;k++){const v=Math.max(0,1-Math.abs((x0+k+.5-c)/fs));wts[o*maxk+k]=v;sum+=v}
    if(!(sum>0)){x0=clamp(Math.floor(c),0,inN-1);n=1;wts[o*maxk]=1;sum=1}
    for(let k=0;k<n;k++)wts[o*maxk+k]/=sum;
    start[o]=x0;cnt[o]=n;
  }
  return{start,cnt,wts,maxk};
}
export function resizeRGB(src,w,h,tw,th){
  const ax=axisWeights(w,tw),ay=axisWeights(h,th),tmp=new Float32Array(h*tw*3);
  for(let y=0;y<h;y++){const row=y*w*4;
    for(let o=0;o<tw;o++){const s=ax.start[o],n=ax.cnt[o],wo=o*ax.maxk;let r=0,g=0,b=0;
      for(let k=0;k<n;k++){const p=row+(s+k)*4,wt=ax.wts[wo+k];r+=src[p]*wt;g+=src[p+1]*wt;b+=src[p+2]*wt}
      const q=(y*tw+o)*3;tmp[q]=r;tmp[q+1]=g;tmp[q+2]=b}}
  const out=new Float32Array(th*tw*3),rowLen=tw*3;
  for(let o=0;o<th;o++){const s=ay.start[o],n=ay.cnt[o],wo=o*ay.maxk;
    for(let x=0;x<rowLen;x++){let v=0;for(let k=0;k<n;k++)v+=tmp[(s+k)*rowLen+x]*ay.wts[wo+k];out[o*rowLen+x]=v}}
  return out;
}

// ---- 幾何 ----
function convexHull(pts){
  pts.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lo=[];for(const p of pts){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p)}
  const up=[];for(let i=pts.length-1;i>=0;i--){const p=pts[i];while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p)}
  lo.pop();up.pop();return lo.concat(up);
}
// 最小外接矩形；回傳 u（接近水平的軸）、v（朝下）、w、h、中心
function minAreaRect(hull){
  let best=null,ba=Infinity;
  const n=hull.length;
  for(let i=0;i<Math.max(n,1);i++){
    let ex=1,ey=0;
    if(n>=2){const a=hull[i],b=hull[(i+1)%n];ex=b[0]-a[0];ey=b[1]-a[1];const l=Math.hypot(ex,ey);if(!l)continue;ex/=l;ey/=l}
    const nx=-ey,ny=ex;let u0=1e18,u1=-1e18,v0=1e18,v1=-1e18;
    for(const p of hull){const pu=p[0]*ex+p[1]*ey,pv=p[0]*nx+p[1]*ny;if(pu<u0)u0=pu;if(pu>u1)u1=pu;if(pv<v0)v0=pv;if(pv>v1)v1=pv}
    const a=(u1-u0)*(v1-v0);
    if(a<ba){ba=a;best={ex,ey,nx,ny,u0,u1,v0,v1}}
    if(n<2)break;
  }
  const{ex,ey,nx,ny,u0,u1,v0,v1}=best,cu=(u0+u1)/2,cv=(v0+v1)/2;
  const cx=cu*ex+cv*nx,cy=cu*ey+cv*ny,a=u1-u0,b=v1-v0;
  let ux,uy,w,h;
  if(Math.abs(ex)>=Math.abs(ey)){ux=ex;uy=ey;w=a;h=b}else{ux=nx;uy=ny;w=b;h=a}
  if(ux<0){ux=-ux;uy=-uy}
  return{cx,cy,ux,uy,vx:-uy,vy:ux,w,h};
}

export class PPOCR{
  constructor(ort,sess,chars){this.ort=ort;this.det=sess.det;this.rec=sess.rec;this.cls=sess.cls;this.chars=chars}
  static async create(ort,{det,rec,cls,dict},sessionOptions={}){
    const mk=m=>m?ort.InferenceSession.create(m,sessionOptions):null;
    const [d,r,c]=await Promise.all([mk(det),mk(rec),mk(cls)]);
    return new PPOCR(ort,{det:d,rec:r,cls:c},dict.split('\n'));
  }
  T(data,dims){return new this.ort.Tensor('float32',data,dims)}

  // ---------- 文字偵測 ----------
  async detect(img,o){
    const{data,width:W,height:H}=img;
    let sc=Math.min(1,o.detMaxSide/Math.max(W,H));
    if(Math.min(W,H)*sc<64)sc=Math.min(4,64/Math.min(W,H));
    const rw=Math.max(32,Math.round(W*sc/32)*32),rh=Math.max(32,Math.round(H*sc/32)*32);
    const rgb=resizeRGB(data,W,H,rw,rh),plane=rw*rh,t=new Float32Array(3*plane);
    for(let c=0;c<3;c++){const ch=o.swapRB?2-c:c,m=MEAN[c],s=STD[c],base=c*plane;
      for(let i=0;i<plane;i++)t[base+i]=(rgb[i*3+ch]/255-m)/s}
    const out=await this.det.run({x:this.T(t,[1,3,rh,rw])});
    const map=out[Object.keys(out)[0]].data;
    return this.boxes(map,rw,rh,W/rw,H/rh,o);
  }
  boxes(map,rw,rh,sx,sy,o){
    const N=rw*rh,lab=new Uint8Array(N),stack=new Int32Array(N),mem=new Int32Array(N),res=[];let mc=0;
    for(let i=0;i<N;i++){
      if(lab[i]||map[i]<=o.thresh)continue;
      const st=mc;let sp=0;stack[sp++]=i;lab[i]=1;mem[mc++]=i;
      while(sp){const p=stack[--sp],x=p%rw,y=(p-x)/rw;
        for(let dy=-1;dy<=1;dy++){const yy=y+dy;if(yy<0||yy>=rh)continue;
          for(let dx=-1;dx<=1;dx++){const xx=x+dx;if(xx<0||xx>=rw)continue;const q=yy*rw+xx;
            if(!lab[q]&&map[q]>o.thresh){lab[q]=1;stack[sp++]=q;mem[mc++]=q}}}}
      if(mc-st<12)continue;
      let y0=1e9,y1=-1;for(let k=st;k<mc;k++){const y=(mem[k]/rw)|0;if(y<y0)y0=y;if(y>y1)y1=y}
      const rows=y1-y0+1,rmin=new Int32Array(rows).fill(1e9),rmax=new Int32Array(rows).fill(-1);
      for(let k=st;k<mc;k++){const p=mem[k],y=(p/rw)|0,x=p-y*rw,r=y-y0;if(x<rmin[r])rmin[r]=x;if(x>rmax[r])rmax[r]=x}
      const pts=[];for(let r=0;r<rows;r++){if(rmax[r]<0)continue;const y=y0+r;pts.push([rmin[r],y],[rmax[r]+1,y],[rmin[r],y+1],[rmax[r]+1,y+1])}
      const rc=minAreaRect(convexHull(pts));
      if(Math.min(rc.w,rc.h)<3)continue;
      // 框內平均機率（同 PaddleOCR 的 box score）
      const hw=rc.w/2,hh=rc.h/2,ex=Math.abs(rc.ux)*hw+Math.abs(rc.vx)*hh,ey=Math.abs(rc.uy)*hw+Math.abs(rc.vy)*hh;
      let sum=0,cnt=0;
      for(let y=Math.max(0,Math.floor(rc.cy-ey));y<Math.min(rh,Math.ceil(rc.cy+ey));y++)
        for(let x=Math.max(0,Math.floor(rc.cx-ex));x<Math.min(rw,Math.ceil(rc.cx+ex));x++){
          const dx=x+.5-rc.cx,dy=y+.5-rc.cy,du=dx*rc.ux+dy*rc.uy,dv=dx*rc.vx+dy*rc.vy;
          if(Math.abs(du)<=hw&&Math.abs(dv)<=hh){sum+=map[y*rw+x];cnt++}}
      if(!cnt||sum/cnt<o.boxThresh)continue;
      // unclip（擴張），再換算回原圖座標
      const d=rc.w*rc.h*o.unclip/(2*(rc.w+rc.h)),W2=rc.w+2*d,H2=rc.h+2*d;
      if(Math.min(W2,H2)<5)continue;
      const P=(a,b)=>[(rc.cx+a*rc.ux*W2/2+b*rc.vx*H2/2)*sx,(rc.cy+a*rc.uy*W2/2+b*rc.vy*H2/2)*sy];
      const TL=P(-1,-1),TR=P(1,-1),BL=P(-1,1);
      const wl=Math.hypot(TR[0]-TL[0],TR[1]-TL[1]),hl=Math.hypot(BL[0]-TL[0],BL[1]-TL[1]);
      res.push({o:TL,u:[(TR[0]-TL[0])/wl,(TR[1]-TL[1])/wl],v:[(BL[0]-TL[0])/hl,(BL[1]-TL[1])/hl],w:wl,h:hl});
    }
    // 由上到下、由左到右（同 PaddleOCR sorted_boxes）
    res.sort((a,b)=>a.o[1]-b.o[1]||a.o[0]-b.o[0]);
    const tol=Math.max(10,0.4*(res.length?res.map(b=>Math.min(b.w,b.h)).sort((a,b)=>a-b)[res.length>>1]:0));
    for(let i=0;i<res.length-1;i++)for(let j=i;j>=0;j--){
      if(Math.abs(res[j+1].o[1]-res[j].o[1])<tol&&res[j+1].o[0]<res[j].o[0]){const t=res[j];res[j]=res[j+1];res[j+1]=t}else break}
    return res;
  }

  // ---------- 取出文字行影像（高 48，閱讀方向朝右） ----------
  crop(img,b,outH=48,maxW=3200){
    const vertical=b.h/b.w>=1.5;
    // 直排：沿 v（由上往下）閱讀，文字行「下方」朝 -u
    const O=vertical?[b.o[0]+b.w*b.u[0],b.o[1]+b.w*b.u[1]]:b.o;
    const a=vertical?b.v:b.u,d=vertical?[-b.u[0],-b.u[1]]:b.v,L=vertical?b.h:b.w,Th=vertical?b.w:b.h;
    const outW=clamp(Math.round(L*outH/Th),1,maxW),kx=L/outW,ky=Th/outH,n=clamp(Math.ceil(Math.max(kx,ky)),1,4);
    const out=new Float32Array(outH*outW*3),{data,width:W,height:H}=img,inv=1/(n*n);
    const fetchBL=(x,y,acc)=>{x=clamp(x-.5,0,W-1);y=clamp(y-.5,0,H-1);const x0=x|0,y0=y|0,x1=Math.min(W-1,x0+1),y1=Math.min(H-1,y0+1),fx=x-x0,fy=y-y0;
      const p00=(y0*W+x0)*4,p10=(y0*W+x1)*4,p01=(y1*W+x0)*4,p11=(y1*W+x1)*4;
      for(let c=0;c<3;c++)acc[c]+=(data[p00+c]*(1-fx)+data[p10+c]*fx)*(1-fy)+(data[p01+c]*(1-fx)+data[p11+c]*fx)*fy};
    const acc=[0,0,0];
    for(let j=0;j<outH;j++)for(let i=0;i<outW;i++){
      acc[0]=acc[1]=acc[2]=0;
      for(let q=0;q<n;q++)for(let p=0;p<n;p++){
        const s=(i+(p+.5)/n)*kx,t=(j+(q+.5)/n)*ky;
        fetchBL(O[0]+s*a[0]+t*d[0],O[1]+s*a[1]+t*d[1],acc)}
      const k=(j*outW+i)*3;out[k]=acc[0]*inv;out[k+1]=acc[1]*inv;out[k+2]=acc[2]*inv}
    const P=(s,t)=>[O[0]+s*L*a[0]+t*Th*d[0],O[1]+s*L*a[1]+t*Th*d[1]];
    return{pix:out,w:outW,h:outH,quad:[P(0,0),P(1,0),P(1,1),P(0,1)],vertical};
  }
  flip(c){
    const{pix,w,h}=c,out=new Float32Array(pix.length);
    for(let j=0;j<h;j++)for(let i=0;i<w;i++){const a=(j*w+i)*3,b=((h-1-j)*w+(w-1-i))*3;out[b]=pix[a];out[b+1]=pix[a+1];out[b+2]=pix[a+2]}
    c.pix=out;const q=c.quad;c.quad=[q[2],q[3],q[0],q[1]];c.flipped=true;
  }

  // ---------- 方向分類（0°／180°） ----------
  async classify(crops,o){
    if(!this.cls||!crops.length)return;
    const W=192,H=48,n=crops.length,t=new Float32Array(n*3*H*W);
    crops.forEach((c,k)=>{
      const rw=Math.min(W,Math.max(1,Math.ceil(H*c.w/c.h)));
      for(let ch=0;ch<3;ch++){const sc=o.swapRB?2-ch:ch;
        for(let j=0;j<H;j++)for(let i=0;i<rw;i++){
          const fx=(i+.5)*c.w/rw-.5,x0=clamp(Math.floor(fx),0,c.w-1),x1=Math.min(c.w-1,x0+1),f=clamp(fx-x0,0,1);
          const v=c.pix[(j*c.w+x0)*3+sc]*(1-f)+c.pix[(j*c.w+x1)*3+sc]*f;
          t[((k*3+ch)*H+j)*W+i]=v/255/.5-1}}});
    const out=await this.cls.run({x:this.T(t,[n,3,H,W])}),r=out[Object.keys(out)[0]].data;
    crops.forEach((c,k)=>{if(r[k*2+1]>r[k*2]&&r[k*2+1]>.9){c.alt={pix:c.pix,w:c.w,h:c.h,quad:c.quad,vertical:c.vertical};this.flip(c)}});
  }

  // ---------- 文字辨識（CTC） ----------
  async recognize_(crops,o,onStep){
    const lines=new Array(crops.length),order=crops.map((_,i)=>i).sort((a,b)=>crops[a].w-crops[b].w);
    const chars=this.chars,BS=o.recBatch;let done=0;
    for(let s=0;s<order.length;s+=BS){
      const ids=order.slice(s,s+BS),H=48,maxW=Math.max(...ids.map(i=>crops[i].w)),B=ids.length;
      const t=new Float32Array(B*3*H*maxW);
      ids.forEach((id,k)=>{const c=crops[id];
        for(let ch=0;ch<3;ch++){const sc=o.swapRB?2-ch:ch;
          for(let j=0;j<H;j++)for(let i=0;i<c.w;i++)t[((k*3+ch)*H+j)*maxW+i]=c.pix[(j*c.w+i)*3+sc]/255/.5-1}});
      const out=await this.rec.run({x:this.T(t,[B,3,H,maxW])}),key=Object.keys(out)[0],dims=out[key].dims,d=out[key].data,T=dims[1],C=dims[2];
      ids.forEach((id,k)=>{
        const c=crops[id],Tv=clamp(Math.ceil(T*c.w/maxW),1,T),em=[];let prev=0;
        for(let tt=0;tt<Tv;tt++){const base=(k*T+tt)*C;let bi=0,bv=-1;for(let q=0;q<C;q++){const v=d[base+q];if(v>bv){bv=v;bi=q}}
          if(bi!==0&&bi!==prev)em.push({ch:chars[bi]||'',p:bv,t:tt});prev=bi}
        lines[id]={em,Tv};
      });
      done+=ids.length;if(onStep)onStep(done/crops.length);
    }
    return lines;
  }

  // ---------- 主流程 ----------
  async pass(img,o,rep){
    rep('det',0);const boxes=o.boxes||await this.detect(img,o);rep('det',1);
    const crops=boxes.map(b=>this.crop(img,b));
    rep('cls',0);await this.classify(crops,o);rep('cls',1);
    // 分類器判定「倒過來」的行：正反兩種都辨識，取信心較高者（避免誤翻）
    const all=[...crops];crops.forEach((c,i)=>{if(c.alt){all.push(Object.assign({pair:i},c.alt));c.pair=-1}});
    const raw=await this.recognize_(all,o,f=>rep('rec',f));
    const out=[];
    crops.forEach((c0,i)=>{
      let c=c0,r=raw[i];
      if(c0.alt){const j=all.findIndex(x=>x.pair===i),ra=raw[j];
        const cf=x=>x&&x.em.length?x.em.reduce((a,e)=>a+e.p,0):0;   // 信心總和（平均值會讓只讀到 1 個字的錯誤方向勝出）
        if(cf(ra)>cf(r)){c=all[j];c.flipped=false;r=ra}else c.flipped=true}
      const{em,Tv}=r;if(!em.length)return;
      const conf=em.reduce((a,e)=>a+e.p,0)/em.length;if(conf<o.dropScore)return;
      const n=em.length,pc=em.map(e=>e.t+.5),g=n>1?(pc[n-1]-pc[0])/(n-1):Tv;
      const bnd=[];bnd.push(Math.max(0,pc[0]-g/2));for(let k=1;k<n;k++)bnd.push((pc[k-1]+pc[k])/2);bnd.push(Math.min(Tv,pc[n-1]+g/2));
      const q=c.quad,L=(s,t)=>{const top=[q[0][0]+(q[1][0]-q[0][0])*s,q[0][1]+(q[1][1]-q[0][1])*s],bot=[q[3][0]+(q[2][0]-q[3][0])*s,q[3][1]+(q[2][1]-q[3][1])*s];
        return[top[0]+(bot[0]-top[0])*t,top[1]+(bot[1]-top[1])*t]};
      const chs=em.map((e,k)=>{const s0=bnd[k]/Tv,s1=bnd[k+1]/Tv,ps=[L(s0,0),L(s1,0),L(s1,1),L(s0,1)];
        return{ch:e.ch,p:e.p,bbox:[Math.min(...ps.map(p=>p[0])),Math.min(...ps.map(p=>p[1])),Math.max(...ps.map(p=>p[0])),Math.max(...ps.map(p=>p[1]))]}});
      const xs=q.map(p=>p[0]),ys=q.map(p=>p[1]);
      out.push({text:em.map(e=>e.ch).join(''),conf,quad:q,vertical:!!c.vertical,flipped:!!c.flipped,chars:chs,
        bbox:[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)]});
    });
    return out;
  }
  // 使用者框選的整塊區域視為「一行」：略過偵測，直接辨識（邊緣以複製像素補一圈，避免字被切邊）
  async asLine(img,o){
    const W=img.width,H=img.height;
    // 邊緣留白多寡會影響手寫／藝術字的讀法：試 4 種，取「信心×字數」最高者
    let res=[],best=-1;
    for(const f of [0,.08,.16,.24]){
      const pad=Math.round(f*Math.min(W,H)),box={o:[-pad,-pad],u:[1,0],v:[0,1],w:W+2*pad,h:H+2*pad};
      const r=await this.pass(img,Object.assign({},o,{boxes:[box]}),()=>{});
      const sc=r.length?r[0].conf*r[0].chars.length:0;
      if(sc>best){best=sc;res=r}
    }
    const cl=(v,m)=>Math.max(0,Math.min(m,v)),cb=b=>[cl(b[0],W),cl(b[1],H),cl(b[2],W),cl(b[3],H)];
    res.forEach(l=>{l.bbox=cb(l.bbox);l.chars.forEach(c=>{c.bbox=cb(c.bbox)})});
    return res;
  }
  async recognize(img,opts={},onProgress){
    const o=Object.assign({detScales:[1280],thresh:.3,boxThresh:.6,unclip:1.5,swapRB:false,recBatch:6,dropScore:.35},opts);
    const res=await this.recognizeDet(img,o,onProgress);
    if(!o.alsoLine)return res;
    // 框選區：若「整塊當一行」的結果明顯更完整（信心×字數更高），就採用它
    const B=await this.asLine(img,o),sc=l=>l.conf*l.chars.length,nA=res.reduce((a,l)=>a+l.chars.length,0),sA=res.reduce((a,l)=>a+sc(l),0);
    if(B.length&&((B[0].conf>=.6&&B[0].chars.length>nA)||(B[0].conf>=.5&&sc(B[0])>sA*1.08)))return B;   // 讀到更多字且信心不低，或總分明顯較高
    return res;
  }
  async recognizeDet(img,o,onProgress){
    const sc=o.detScales,all=[];
    for(let k=0;k<sc.length;k++){
      const W={det:[0,.3],cls:[.3,.35],rec:[.35,1]};
      const rep=(s,f)=>{if(onProgress)onProgress(s,(k+W[s][0]+f*(W[s][1]-W[s][0]))/sc.length)};
      all.push(...await this.pass(img,Object.assign({},o,{detMaxSide:sc[k]}),rep));
    }
    return sc.length>1?mergeLines(all):sortLines(all);
  }
}

// 多尺度結果合併：重疊的行只留「信心×字數」最高者（偏好較完整的行）
function mergeLines(lines){
  const area=b=>Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  const inter=(a,b)=>area([Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])]);
  const sorted=[...lines].sort((a,b)=>b.conf*b.chars.length-a.conf*a.chars.length),keep=[];
  for(const l of sorted){
    if(keep.some(k=>inter(k.bbox,l.bbox)/Math.max(1,Math.min(area(k.bbox),area(l.bbox)))>.5))continue;
    keep.push(l);
  }
  return sortLines(keep);
}
function sortLines(ls){
  ls.sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0]);
  const hs=ls.map(l=>Math.min(l.bbox[2]-l.bbox[0],l.bbox[3]-l.bbox[1])).sort((a,b)=>a-b),tol=Math.max(10,.4*(hs[hs.length>>1]||0));
  for(let i=0;i<ls.length-1;i++)for(let j=i;j>=0;j--){
    if(Math.abs(ls[j+1].bbox[1]-ls[j].bbox[1])<tol&&ls[j+1].bbox[0]<ls[j].bbox[0]){const t=ls[j];ls[j]=ls[j+1];ls[j+1]=t}else break}
  return ls;
}
