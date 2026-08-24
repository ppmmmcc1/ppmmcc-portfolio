/* =====================================================================
   render.js — renders project/concept lists and the project detail page.
   Source of truth: Supabase (published rows), read live with the public
   anon key so anything you publish in /admin.html appears on the site
   immediately. If Supabase is unreachable (e.g. a free project paused)
   or not configured, it falls back to the committed data/projects.json,
   so the public site never goes blank. Run `npm run sync` to refresh
   that offline snapshot and localize images when you want a durable copy.
   ===================================================================== */
(function () {
    'use strict';

    var JSON_FALLBACK = 'data/projects.json';

    function supaCfg() {
        var c = window.SUPA_CONFIG;
        if (!c || !c.url || !c.anonKey) { return null; }
        if (c.url.indexOf('YOUR-PROJECT') !== -1 || c.anonKey.indexOf('YOUR-ANON') !== -1) { return null; }
        return c;
    }
    function restUrl(cfg, query) {
        return cfg.url.replace(/\/+$/, '') + '/rest/v1/projects?' + query;
    }
    function supaHeaders(cfg) {
        return { apikey: cfg.anonKey, Authorization: 'Bearer ' + cfg.anonKey };
    }

    document.addEventListener('DOMContentLoaded', function () {
        var list = document.querySelector('[data-work-list]');
        var detail = document.getElementById('project-detail');
        if (list) { initList(list); }
        if (detail) { initDetail(detail); }
    });

    /* ---------------- data access ---------------- */

    function fetchAll() {
        // Live-first: read published rows straight from Supabase so new
        // projects appear the moment they're published. Fall back to the
        // committed snapshot if Supabase isn't reachable or configured.
        return fetchLive().then(function (rows) {
            return (rows && rows.length) ? rows : fetchJson();
        });
    }

    function fetchLive() {
        var cfg = supaCfg();
        if (!cfg) { return Promise.resolve(null); }
        // Row-Level Security limits the anon key to published rows only.
        return fetch(restUrl(cfg, 'select=*&order=sort_order'), { headers: supaHeaders(cfg), cache: 'no-cache' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (rows) {
                if (!Array.isArray(rows)) { return null; }
                return rows.filter(function (x) { return x.published !== false; });
            })
            .catch(function () { return null; });
    }

    function fetchJson() {
        return fetch(JSON_FALLBACK, { cache: 'no-cache' })
            .then(function (r) { return r.ok ? r.json() : []; })
            .then(function (rows) { return (rows || []).filter(function (r) { return r.published !== false; }); })
            .catch(function () { return []; });
    }

    /* ---------------- list pages ---------------- */

    function initList(container) {
        var status = container.getAttribute('data-work-list'); // 'project' | 'concept'
        container.innerHTML = '<p class="state-msg">Loading…</p>';

        fetchAll().then(function (rows) {
            var items = rows
                .filter(function (r) { return r.status === status; })
                .sort(function (a, b) { return (a.sort_order || 0) - (b.sort_order || 0); });

            if (!items.length) {
                container.innerHTML = '<p class="state-msg">Nothing here yet, check back soon.</p>';
                return;
            }
            container.classList.add('work-grid');
            container.innerHTML = items.map(cardHtml).join('');
            reveal(container.querySelectorAll('.work-card'));
        });
    }

    function cardHtml(p, i) {
        var num = String(i + 1).padStart(2, '0');
        var img = p.hero_image
            ? '<img src="' + attr(p.hero_image) + '" alt="' + attr(p.title) + '" loading="lazy">'
            : '';
        var flag = p.status === 'concept'
            ? '<span class="status-flag concept">Concept</span>'
            : '<span class="status-flag">Built</span>';
        var tags = (p.tags || []).slice(0, 4).map(function (t) {
            return '<span class="tag">' + esc(t) + '</span>';
        }).join('');
        return '' +
            '<a class="work-card" href="project.html?slug=' + encodeURIComponent(p.slug) + '" data-reveal>' +
                '<div class="frame">' +
                    '<span class="num">' + num + '</span>' + flag + img +
                '</div>' +
                '<div class="work-meta">' +
                    '<h3 class="work-title">' + esc(p.title) + '</h3>' +
                    (tags ? '<div class="work-tags">' + tags + '</div>' : '') +
                '</div>' +
            '</a>';
    }

    /* ---------------- detail page ---------------- */

    function initDetail(root) {
        var slug = new URLSearchParams(location.search).get('slug');
        if (!slug) { root.innerHTML = notFound(); return; }

        fetchOne(slug).then(function (p) {
            if (!p) { root.innerHTML = notFound(); return; }
            document.title = p.title + ' — Phillip Castro';
            setMeta('description', p.summary || p.subtitle || p.title);
            root.innerHTML = detailHtml(p);
            reveal(root.querySelectorAll('[data-reveal]'));
            if (window.mountCadViewers) { window.mountCadViewers(); }
        });
    }

    function fetchOne(slug) {
        // Live-first (see fetchAll), falling back to the committed snapshot.
        return fetchLiveOne(slug).then(function (p) {
            return p || fetchJsonOne(slug);
        });
    }

    function fetchLiveOne(slug) {
        var cfg = supaCfg();
        if (!cfg) { return Promise.resolve(null); }
        var q = 'select=*&slug=eq.' + encodeURIComponent(slug);
        return fetch(restUrl(cfg, q), { headers: supaHeaders(cfg), cache: 'no-cache' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (rows) {
                // RLS hides drafts from the anon key, so this is a published row or nothing.
                return (Array.isArray(rows) && rows[0] && rows[0].published !== false) ? rows[0] : null;
            })
            .catch(function () { return null; });
    }

    function fetchJsonOne(slug) {
        return fetchJson().then(function (rows) {
            return rows.filter(function (r) { return r.slug === slug; })[0] || null;
        });
    }

    function detailHtml(p) {
        var back = p.status === 'concept' ? 'concepts.html' : 'projects.html';
        var backLabel = p.status === 'concept' ? 'Concepts' : 'Projects';
        var tags = (p.tags || []).map(function (t) { return '<span class="tag">' + esc(t) + '</span>'; }).join('');
        var specs = (p.specs || []).map(function (s) {
            return '<div class="spec"><div class="spec-label">' + esc(s.label) + '</div>' +
                   '<div class="spec-value">' + esc(s.value) + '</div></div>';
        }).join('');

        var labels = p.status === 'concept'
            ? ['The concept', 'Technical approach', 'Impact vision']
            : ['Why I built it', 'How I built it', 'What I overcame'];
        var blocks = [
            block('01', labels[0], p.why_md),
            block('02', labels[1], p.how_md),
            block('03', labels[2], p.problems_md)
        ].join('');

        var cad = p.stl_url
            ? cadHtml(p.stl_url, p.title)
            : '';

        // Fold the hero in as the first gallery image so the bottom of the
        // page reads: 3D model → gallery.
        var media = [];
        if (p.hero_image) { media.push({ url: p.hero_image, alt: p.title }); }
        media = media.concat(p.gallery || []);
        var gallery = galleryHtml(media);

        // Order for every project/concept: header → 3D model → story text →
        // pictures (gallery, hero folded in) grouped at the bottom.
        return '' +
            '<div class="container">' +
                '<header class="detail-head" data-reveal>' +
                    '<a class="back-link" href="' + back + '">' +
                        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5M11 18l-6-6 6-6"/></svg> ' +
                        backLabel +
                    '</a>' +
                    '<h1 class="detail-title">' + esc(p.title) + '</h1>' +
                    (p.subtitle ? '<p class="detail-subtitle">' + esc(p.subtitle) + '</p>' : '') +
                    (tags ? '<div class="detail-tags">' + tags + '</div>' : '') +
                    (specs ? '<div class="specs-bar">' + specs + '</div>' : '') +
                '</header>' +
                cad +
                '<div class="narrative">' + blocks + '</div>' +
                gallery +
            '</div>';
    }

    function block(n, label, md) {
        if (!md || !md.trim()) { return ''; }
        return '<section class="block" data-reveal>' +
            '<p class="block-label"><span class="n">' + n + '</span> ' + esc(label) + '</p>' +
            '<div class="prose">' + mdToHtml(md) + '</div>' +
        '</section>';
    }

    function galleryHtml(gallery) {
        var g = gallery || [];
        if (!g.length) { return ''; }
        var cls = g.length === 1 ? 'gallery single' : 'gallery';
        var figs = g.map(function (it) {
            return '<figure data-reveal><img src="' + attr(it.url) + '" alt="' + attr(it.alt || '') + '" loading="lazy"></figure>';
        }).join('');
        return '<section class="gallery-block" data-reveal>' +
            '<h2 class="gallery-title"><span class="dot"></span>Gallery</h2>' +
            '<div class="' + cls + '">' + figs + '</div>' +
        '</section>';
    }

    function cadHtml(url, title) {
        return '' +
        '<div class="cad" data-stl-url="' + attr(url) + '" data-reveal>' +
            '<div class="cad-bar">' +
                '<span class="cad-name"><span class="dot"></span> 3D Model — ' + esc(title) + '</span>' +
                '<div class="cad-controls">' +
                    '<button class="cad-btn" data-cad="rotate" aria-pressed="true">Auto-rotate</button>' +
                    '<button class="cad-btn" data-cad="wire">Wireframe</button>' +
                    '<button class="cad-btn" data-cad="reset">Reset</button>' +
                    '<button class="cad-btn" data-cad="full">Fullscreen</button>' +
                '</div>' +
            '</div>' +
            '<div class="cad-stage">' +
                '<div class="cad-overlay"><span><span class="cad-spinner"></span>Loading model…</span></div>' +
            '</div>' +
            '<div class="cad-hint">Drag to orbit · scroll to zoom · right-drag to pan</div>' +
        '</div>';
    }

    function notFound() {
        return '<div class="container"><div class="detail-head">' +
            '<a class="back-link" href="projects.html">← Projects</a>' +
            '<h1 class="detail-title">Not found</h1>' +
            '<p class="detail-subtitle">That project doesn’t exist or hasn’t been published yet.</p>' +
            '</div></div>';
    }

    /* ---------------- helpers ---------------- */

    function reveal(nodes) {
        // If GSAP reveals already ran, [data-reveal] items are hidden by
        // motion.js. For content injected after load, make sure it shows.
        if (!window.gsap) { return; }
        try {
            window.gsap.set(nodes, { opacity: 1, y: 0, clearProps: 'opacity,transform' });
            if (window.ScrollTrigger) { window.ScrollTrigger.refresh(); }
        } catch (e) {}
    }

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    function attr(s) { return esc(s).replace(/"/g, '&quot;'); }

    // markdown-lite: paragraphs, **bold**, and "- " bullet lists
    function mdToHtml(md) {
        var blocks = String(md).replace(/\r\n/g, '\n').split(/\n{2,}/);
        return blocks.map(function (b) {
            var lines = b.split('\n');
            var isList = lines.every(function (l) { return /^\s*[-*]\s+/.test(l) || !l.trim(); });
            if (isList && /[-*]\s+/.test(b)) {
                var lis = lines.filter(function (l) { return l.trim(); })
                    .map(function (l) { return '<li>' + inline(l.replace(/^\s*[-*]\s+/, '')) + '</li>'; });
                return '<ul>' + lis.join('') + '</ul>';
            }
            return '<p>' + inline(b.replace(/\n/g, ' ')) + '</p>';
        }).join('');
    }
    function inline(s) {
        return esc(s).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    }

    function setMeta(name, content) {
        var el = document.querySelector('meta[name="' + name + '"]');
        if (el) { el.setAttribute('content', content); }
    }
})();
