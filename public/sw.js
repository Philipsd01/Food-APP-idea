// Caches only the static app shell — index.html, style.css, manifest, icon.
// Every real request (/search, /nearby, /photo, third-party CDN scripts,
// map tiles) always goes straight to the network, so results stay live.
// This exists solely so the app is installable and doesn't show a blank
// white screen if the shell itself is requested while offline or on a
// flaky connection — not a general offline mode for the app's data.
const CACHE_NAME = 'dine-shell-v1';
const SHELL_ASSETS = ['/', '/style.css', '/manifest.json', '/icon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || !SHELL_ASSETS.includes(url.pathname)) return;

  event.respondWith(
    fetch(req)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
        return res;
      })
      .catch(() => caches.match(req))
  );
});
