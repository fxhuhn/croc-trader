const CACHE_NAME = 'croc-trader-v1';

self.addEventListener('install', (event) => {
    // Basic service worker pass-through
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
    // Network-first or purely network pass-through
    event.respondWith(fetch(event.request));
});
