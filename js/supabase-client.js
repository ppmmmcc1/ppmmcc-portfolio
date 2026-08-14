/* =====================================================================
   supabase-client.js — creates the shared Supabase client.
   Loads AFTER the supabase-js UMD bundle and config.js.

   Exposes:
     window.sb         -> the Supabase client (or null if not configured)
     window.SUPA_READY -> boolean, true when a real client was created
     window.SUPA_CONFIG-> the config object (buckets, etc.)
   ===================================================================== */
(function () {
    'use strict';
    var cfg = window.SUPA_CONFIG || {};
    window.sb = null;
    window.SUPA_READY = false;

    function configured() {
        return !!(cfg.url && cfg.anonKey &&
            cfg.url.indexOf('YOUR-PROJECT') === -1 &&
            cfg.anonKey.indexOf('YOUR-ANON') === -1);
    }

    if (typeof window.supabase === 'undefined' || !window.supabase.createClient) {
        console.warn('[supabase] library not loaded.');
        return;
    }
    if (!configured()) {
        console.warn('[supabase] not configured yet — edit js/config.js with your project URL and anon key.');
        return;
    }

    // autoRefreshToken is intentionally OFF. This admin page is opened
    // rarely, and a background refresh timer pointed at a paused/removed
    // project retries forever — that's what produced the browser's
    // "too many attempts fetching resources" error. The session still
    // persists; if the token expires the owner simply signs in again.
    window.sb = window.supabase.createClient(cfg.url, cfg.anonKey, {
        auth: { persistSession: true, autoRefreshToken: false }
    });
    window.SUPA_READY = true;
})();
