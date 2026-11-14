// 改了任何檔案後，請把 VERSION 加 1，使用者就會看到「有新版本可用」
const VERSION = 'v18';
const CACHE = 'ocr-' + VERSION;
const ASSETS = [
  './', 'index.html', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/maskable-512.png', 'icons/apple-touch-icon.png',
  'vendor/tesseract.min.js', 'vendor/worker.min.js',
  'vendor/core/tesseract-core-lstm.wasm.js', 'vendor/core/tesseract-core-simd-lstm.wasm.js', 'vendor/core/tesseract-core-relaxedsimd-lstm.wasm.js',
  'vendor/lang/chi_tra.traineddata.gz', 'vendor/lang/eng.traineddata.gz',
  'vendor/paddle/pp-ocr.mjs', 'vendor/paddle/worker.mjs'   // 小型程式碼隨版本更新；大型模型／執行環境才用下方的長期快取
];

self.addEventListener('install', e => {
  // 不自動 skipWaiting：讓使用者按「立即更新」才切換，避免使用中途被換版
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k.startsWith('ocr-') && k !== CACHE).map(k => caches.delete(k))))
      .then(() => caches.open('paddle-v1').then(async c => {   // 舊版曾把程式碼也放進長期快取，清掉以免讀到舊檔
        for (const r of await c.keys()) if (/\/(pp-ocr|worker)\.mjs$|NOTICE\.md$/.test(r.url)) await c.delete(r);
      }).catch(() => {}))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', e => { if (e.data === 'SKIP_WAITING') self.skipWaiting(); });

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  // PaddleOCR 模型與執行環境（約 37MB）：第一次使用時才下載，之後快取（不隨 VERSION 清除）
  // 只有大型、不會變的檔案（模型、字典、onnxruntime）走長期快取
  if (/\/vendor\/paddle\/(ort[^/]*\.mjs|[^/]*\.(wasm|onnx|ort|txt))$/.test(new URL(req.url).pathname)) {
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
      if (res.ok && res.status === 200) { const cp = res.clone(); caches.open('paddle-v1').then(c => c.put(req, cp)); }
      return res;
    })));
    return;
  }
  e.respondWith(
    caches.match(req, { ignoreSearch: true }).then(hit => {
      if (hit) return hit;
      return fetch(req).catch(() =>
        req.mode === 'navigate' ? caches.match('index.html') : Response.error()
      );
    })
  );
});
