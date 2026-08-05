/* =====================================================================
   render.js — renders project/concept lists and the project detail page.
   Source of truth: Supabase (published rows). Falls back to the bundled
   data/projects.json so the public site looks complete even before
   Supabase is configured. Live admin edits override the JSON the moment
   the database has rows.
   ===================================================================== */
(function () {
    'use strict';

    var JSON_FALLBACK = 'data/projects.json';

    document.addEventListener('DOMContentLoaded', function () {
        var list = document.querySelector('[data-work-list]');
        var detail = document.getElementById('project-detail');
        if (list) { initList(list); }
        if (detail) { initDetail(detail); }
    });

    /* ---------------- data access ---------------- */

    function fetchAll() {
        // Local-first: the public site reads the committed snapshot in
        // data/projects.json and never calls Supabase. Author projects in
        // /admin.html, then run `npm run sync` to refresh this file.
        return fetchJson();
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
        // Local-first (see fetchAll): served from data/projects.json.
        return fetchJsonOne(slug);
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

        var hero = p.hero_image
            ? '<div class="detail-hero" data-reveal><img src="' + attr(p.hero_image) + '" alt="' + attr(p.title) + '"></div>'
            : '';

        var cad = p.stl_url
            ? cadHtml(p.stl_url, p.title)
            : '';

        var gallery = galleryHtml(p.gallery);

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
                hero +
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
        return '<div class="' + cls + '">' + figs + '</div>';
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
