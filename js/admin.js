/* =====================================================================
   admin.js — private dashboard for managing projects.
   Auth + reads/writes go through Supabase. The page being unlinked is
   only convenience; Row-Level Security is the real guard, so even if
   someone opens admin.html they can't read drafts or write anything
   without logging in as the owner.
   ===================================================================== */
(function () {
    'use strict';

    var root, bar, toastEl;
    var BUCKET_IMG = (window.SUPA_CONFIG && window.SUPA_CONFIG.imagesBucket) || 'images';
    var BUCKET_STL = (window.SUPA_CONFIG && window.SUPA_CONFIG.modelsBucket) || 'models';

    // transient state for the editor's uploaded media
    var ed = blankMedia();
    function blankMedia() { return { hero_image: '', gallery: [], stl_url: '' }; }

    document.addEventListener('DOMContentLoaded', function () {
        root = document.getElementById('admin-root');
        bar = document.getElementById('bar-actions');
        toastEl = document.getElementById('toast');

        if (!window.SUPA_READY || !window.sb) { renderSetupNeeded(); return; }

        window.sb.auth.getSession().then(function (res) {
            route(res.data.session ? res.data.session.user : null);
        });
        window.sb.auth.onAuthStateChange(function (_evt, session) {
            route(session ? session.user : null);
        });
    });

    function route(user) {
        if (user) { setBarLoggedIn(); renderDashboard(); }
        else { setBarLoggedOut(); renderLogin(); }
    }

    /* ---------------- top bar ---------------- */
    function setBarLoggedIn() {
        bar.innerHTML =
            '<a class="btn btn-sm" href="index.html"><span>View site ↗</span></a>' +
            '<button class="btn btn-sm" id="logout-btn"><span>Log out</span></button>';
        document.getElementById('logout-btn').addEventListener('click', function () {
            window.sb.auth.signOut();
        });
    }
    function setBarLoggedOut() {
        bar.innerHTML = '<a class="btn btn-sm" href="index.html"><span>View site ↗</span></a>';
    }

    /* ---------------- setup-needed ---------------- */
    function renderSetupNeeded() {
        root.innerHTML =
            '<div class="login-wrap"><div class="login-card">' +
            '<h1>Almost there</h1>' +
            '<p class="sub">Supabase isn’t connected yet. Follow <code>supabase/README.md</code>, then paste your project URL and anon key into <code>js/config.js</code>.</p>' +
            '<a class="btn btn-primary" href="index.html"><span>Back to site</span></a>' +
            '</div></div>';
    }

    /* ---------------- login ---------------- */
    function renderLogin() {
        root.innerHTML =
            '<div class="login-wrap"><form class="login-card" id="login-form">' +
                '<h1>Sign in</h1>' +
                '<p class="sub">Owner access only.</p>' +
                '<div class="field"><label for="email">Email</label>' +
                    '<input class="input" type="email" id="email" autocomplete="username" required></div>' +
                '<div class="field"><label for="password">Password</label>' +
                    '<input class="input" type="password" id="password" autocomplete="current-password" required></div>' +
                '<button class="btn btn-primary" type="submit" id="signin-btn" style="width:100%;justify-content:center"><span>Sign in</span></button>' +
                '<p class="login-error" id="login-error" role="alert" hidden></p>' +
            '</form></div>';

        var btn = document.getElementById('signin-btn');
        var errBox = document.getElementById('login-error');

        function showLoginError(msg) {
            if (!msg) { errBox.hidden = true; errBox.textContent = ''; return; }
            errBox.hidden = false; errBox.textContent = msg;
            toast(msg, true);
        }

        document.getElementById('login-form').addEventListener('submit', function (e) {
            e.preventDefault();
            var email = val('email').trim(), pw = val('password');
            showLoginError('');
            setBtnBusy(btn, true);

            window.sb.auth.signInWithPassword({ email: email, password: pw })
                .then(function (res) {
                    setBtnBusy(btn, false);
                    if (res.error) { showLoginError(authErrorMessage(res.error)); }
                    // success handled by onAuthStateChange
                })
                .catch(function (err) {
                    // signInWithPassword rejects on network/CORS/library failures,
                    // which otherwise show nothing at all.
                    setBtnBusy(btn, false);
                    showLoginError(authErrorMessage(err));
                });
        });
    }

    // Turn a raw Supabase auth error into a plain, actionable message so the
    // login screen explains what to fix instead of dumping a scary string.
    function authErrorMessage(error) {
        var msg = (error && error.message) ? String(error.message) : '';
        var code = (error && error.code) ? String(error.code) : '';
        var status = error ? (error.status || error.statusCode || 0) : 0;
        var low = msg.toLowerCase();

        // No HTTP response came back — network, CORS, or a bad URL/anon key.
        if (!status && (low.indexOf('failed to fetch') !== -1 ||
                        low.indexOf('networkerror') !== -1 ||
                        low.indexOf('load failed') !== -1 ||
                        low.indexOf('fetch') !== -1)) {
            return 'Couldn’t reach the auth server. Check your connection, and that the Supabase URL and anon key in js/config.js are correct.';
        }
        if (code === 'email_not_confirmed' || low.indexOf('email not confirmed') !== -1) {
            return 'This account’s email isn’t confirmed yet. In Supabase → Authentication → Users, open your user and confirm it — or delete it and re-create it with “Auto confirm” ticked.';
        }
        if (code === 'invalid_credentials' || low.indexOf('invalid login credentials') !== -1) {
            return 'Email or password is incorrect. Use the exact email the account was created with in Supabase → Authentication → Users, and reset the password there if unsure.';
        }
        if (code === 'over_request_rate_limit' || status === 429 || low.indexOf('rate limit') !== -1) {
            return 'Too many attempts. Wait a minute, then try again.';
        }
        // 5xx (and Supabase’s "Database error querying schema" 500) are project-side.
        if ((status && status >= 500) || low.indexOf('database error') !== -1 || low.indexOf('unexpected') !== -1) {
            return 'Supabase hit a server error' + (status ? ' (' + status + ')' : '') +
                (msg ? ' — “' + msg + '”' : '') +
                '. This is a project-side issue, not your password. Check Authentication → Logs in the Supabase dashboard.';
        }
        return msg || 'Sign-in failed. Please try again.';
    }

    function setBtnBusy(btn, busy) {
        if (!btn) { return; }
        btn.disabled = busy;
        var span = btn.querySelector('span');
        if (!span) { return; }
        if (busy) { btn.dataset.label = span.textContent; span.textContent = 'Signing in…'; }
        else if (btn.dataset.label) { span.textContent = btn.dataset.label; }
    }

    /* ---------------- dashboard ---------------- */
    function renderDashboard() {
        root.innerHTML =
            '<div class="dash-head">' +
                '<h1>Projects</h1>' +
                '<div class="pr-actions">' +
                    '<button class="btn" id="import-btn"><span>Import starter content</span></button>' +
                    '<button class="btn" id="export-btn"><span>Export JSON</span></button>' +
                    '<button class="btn btn-primary" id="new-btn"><span>+ New project</span></button>' +
                '</div>' +
            '</div>' +
            '<div id="rows"><p class="state-msg">Loading…</p></div>';

        document.getElementById('new-btn').addEventListener('click', function () { renderEditor(null); });
        document.getElementById('export-btn').addEventListener('click', exportJson);
        document.getElementById('import-btn').addEventListener('click', importSeed);

        loadRows();
    }

    function loadRows() {
        window.sb.from('projects').select('*')
            .order('status', { ascending: true })
            .order('sort_order', { ascending: true })
            .then(function (res) {
                var box = document.getElementById('rows');
                if (!box) { return; }
                if (res.error) { box.innerHTML = '<p class="state-msg">' + esc(res.error.message) + '</p>'; return; }
                var rows = res.data || [];
                if (!rows.length) {
                    box.innerHTML = '<p class="state-msg">No projects yet. Create one, or import the starter content.</p>';
                    return;
                }
                box.className = 'proj-rows';
                box.innerHTML = rows.map(rowHtml).join('');
                rows.forEach(wireRow);
            });
    }

    function rowHtml(p) {
        var badge = p.published ? '<span class="badge live">Live</span>' : '<span class="badge draft">Draft</span>';
        return '<div class="proj-row" data-id="' + p.id + '">' +
            '<div><div class="pr-title">' + esc(p.title) + '</div>' +
                '<div class="pr-meta">' + esc(p.status) + ' · ' + esc(p.slug) + '</div></div>' +
            badge +
            '<button class="btn btn-sm" data-act="toggle"><span>' + (p.published ? 'Unpublish' : 'Publish') + '</span></button>' +
            '<div class="pr-actions">' +
                '<button class="btn btn-sm" data-act="edit"><span>Edit</span></button>' +
                '<button class="btn btn-sm btn-danger" data-act="del"><span>Delete</span></button>' +
            '</div>' +
        '</div>';
    }

    function wireRow(p) {
        var row = document.querySelector('.proj-row[data-id="' + p.id + '"]');
        if (!row) { return; }
        row.querySelector('[data-act="edit"]').addEventListener('click', function () { renderEditor(p); });
        row.querySelector('[data-act="toggle"]').addEventListener('click', function () {
            window.sb.from('projects').update({ published: !p.published }).eq('id', p.id).then(function (res) {
                if (res.error) { toast(res.error.message, true); } else { toast(p.published ? 'Unpublished' : 'Published'); loadRows(); }
            });
        });
        row.querySelector('[data-act="del"]').addEventListener('click', function () {
            if (!confirm('Delete “' + p.title + '”? This cannot be undone.')) { return; }
            window.sb.from('projects').delete().eq('id', p.id).then(function (res) {
                if (res.error) { toast(res.error.message, true); } else { toast('Deleted'); loadRows(); }
            });
        });
    }

    /* ---------------- editor ---------------- */
    function renderEditor(p) {
        var isNew = !p;
        p = p || { status: 'project', tags: [], gallery: [], specs: [], published: false, sort_order: 0 };
        ed = { hero_image: p.hero_image || '', gallery: (p.gallery || []).slice(), stl_url: p.stl_url || '' };

        root.innerHTML =
            '<div class="dash-head"><h1>' + (isNew ? 'New project' : 'Edit project') + '</h1>' +
                '<button class="btn btn-sm" id="back-btn"><span>← Back</span></button></div>' +
            '<form id="editor-form" class="editor">' +
                '<div class="panel">' +
                    '<h2>Story</h2>' +
                    field('Title', '<input class="input" id="f-title" value="' + attr(p.title) + '" required>') +
                    field('Slug (URL)', '<input class="input" id="f-slug" value="' + attr(p.slug) + '" placeholder="auto from title">', 'Lowercase, words-with-dashes. Leave blank to auto-generate.') +
                    field('Subtitle', '<input class="input" id="f-subtitle" value="' + attr(p.subtitle) + '">') +
                    field('Summary (card + meta)', '<textarea class="textarea" id="f-summary" style="min-height:70px">' + esc(p.summary) + '</textarea>') +
                    field(labelFor(p.status, 0), '<textarea class="textarea" id="f-why">' + esc(p.why_md) + '</textarea>') +
                    field(labelFor(p.status, 1), '<textarea class="textarea" id="f-how">' + esc(p.how_md) + '</textarea>') +
                    field(labelFor(p.status, 2), '<textarea class="textarea" id="f-problems">' + esc(p.problems_md) + '</textarea>') +
                '</div>' +
                '<div class="panel">' +
                    '<h2>Settings</h2>' +
                    '<div class="row-2">' +
                        field('Type', '<select class="select" id="f-status">' +
                            '<option value="project"' + (p.status === 'project' ? ' selected' : '') + '>Project (built)</option>' +
                            '<option value="concept"' + (p.status === 'concept' ? ' selected' : '') + '>Concept</option></select>') +
                        field('Sort order', '<input class="input" type="number" id="f-sort" value="' + (p.sort_order || 0) + '">') +
                    '</div>' +
                    field('Tags (comma-separated)', '<input class="input" id="f-tags" value="' + attr((p.tags || []).join(', ')) + '">') +
                    '<div class="field"><span class="field-label">Specs</span>' +
                        '<div class="spec-rows" id="spec-rows"></div>' +
                        '<button type="button" class="btn btn-sm" id="add-spec" style="margin-top:8px"><span>+ Add spec</span></button>' +
                    '</div>' +
                    '<div class="checkbox-row" style="margin-bottom:1.5rem"><input type="checkbox" id="f-pub"' + (p.published ? ' checked' : '') + '><label for="f-pub" style="text-transform:none;letter-spacing:0">Published (visible on the site)</label></div>' +

                    '<h2>Media</h2>' +
                    '<div class="field"><span class="field-label">Hero image</span>' +
                        '<div class="dropzone" id="hero-drop">Click or drop an image</div>' +
                        '<div class="thumbs" id="hero-thumb"></div></div>' +
                    '<div class="field"><span class="field-label">Gallery images</span>' +
                        '<div class="dropzone" id="gal-drop">Click or drop images</div>' +
                        '<div class="thumbs" id="gal-thumbs"></div></div>' +
                    '<div class="field"><span class="field-label">3D model (optional)</span>' +
                        '<div class="dropzone" id="stl-drop">Click or drop a model — STL, GLB, glTF, OBJ, 3MF</div>' +
                        '<span class="hint">SolidWorks: Save As → GLB for assemblies, or STL for single parts. Native .SLDPRT/.SLDASM aren’t supported by browsers.</span>' +
                        '<div id="stl-pill"></div></div>' +
                '</div>' +
                '<div class="editor-actions">' +
                    '<button type="submit" class="btn btn-primary"><span>' + (isNew ? 'Create project' : 'Save changes') + '</span></button>' +
                    '<button type="button" class="btn" id="cancel-btn"><span>Cancel</span></button>' +
                '</div>' +
            '</form>';

        document.getElementById('back-btn').addEventListener('click', renderDashboard);
        document.getElementById('cancel-btn').addEventListener('click', renderDashboard);

        // auto-slug from title when slug is empty
        var titleEl = document.getElementById('f-title');
        var slugEl = document.getElementById('f-slug');
        titleEl.addEventListener('input', function () {
            if (!slugEl.dataset.touched) { slugEl.value = slugify(titleEl.value); }
        });
        slugEl.addEventListener('input', function () { slugEl.dataset.touched = '1'; });

        // specs
        (p.specs && p.specs.length ? p.specs : []).forEach(addSpecRow);
        document.getElementById('add-spec').addEventListener('click', function () { addSpecRow({ label: '', value: '' }); });

        // uploads
        wireDrop('hero-drop', 'image/*', false, function (files) { uploadImages(files.slice(0, 1), BUCKET_IMG, function (urls) { ed.hero_image = urls[0]; renderHeroThumb(); }); });
        wireDrop('gal-drop', 'image/*', true, function (files) { uploadImages(files, BUCKET_IMG, function (urls) { urls.forEach(function (u) { ed.gallery.push({ url: u, alt: '' }); }); renderGalThumbs(); }); });
        wireDrop('stl-drop', '.stl,.glb,.gltf,.obj,.3mf,model/stl,model/gltf-binary', false, function (files) { uploadStl(files[0]); });

        renderHeroThumb(); renderGalThumbs(); renderStlPill();

        document.getElementById('editor-form').addEventListener('submit', function (e) {
            e.preventDefault();
            saveProject(p.id || null);
        });
    }

    function labelFor(status, i) {
        return status === 'concept'
            ? ['The concept', 'Technical approach', 'Impact vision'][i]
            : ['Why I built it', 'How I built it', 'What I overcame'][i];
    }

    function addSpecRow(spec) {
        var box = document.getElementById('spec-rows');
        var div = document.createElement('div');
        div.className = 'spec-row';
        div.innerHTML =
            '<input class="input spec-label-in" placeholder="Label" value="' + attr(spec.label) + '">' +
            '<input class="input spec-value-in" placeholder="Value" value="' + attr(spec.value) + '">' +
            '<button type="button" class="icon-btn" aria-label="Remove">×</button>';
        div.querySelector('.icon-btn').addEventListener('click', function () { div.remove(); });
        box.appendChild(div);
    }

    function renderHeroThumb() {
        var box = document.getElementById('hero-thumb');
        box.innerHTML = ed.hero_image ? thumbHtml(ed.hero_image, 'hero') : '';
        if (ed.hero_image) { box.querySelector('button').addEventListener('click', function () { ed.hero_image = ''; renderHeroThumb(); }); }
    }
    function renderGalThumbs() {
        var box = document.getElementById('gal-thumbs');
        box.innerHTML = ed.gallery.map(function (g, i) { return thumbHtml(g.url, i); }).join('');
        Array.prototype.forEach.call(box.querySelectorAll('button'), function (b) {
            b.addEventListener('click', function () { ed.gallery.splice(parseInt(b.dataset.i, 10), 1); renderGalThumbs(); });
        });
    }
    function renderStlPill() {
        var box = document.getElementById('stl-pill');
        if (!ed.stl_url) { box.innerHTML = ''; return; }
        box.innerHTML = '<span class="file-pill">' + esc(fileName(ed.stl_url)) + ' <button type="button">remove</button></span>';
        box.querySelector('button').addEventListener('click', function () { ed.stl_url = ''; renderStlPill(); });
    }
    function thumbHtml(url, i) {
        return '<div class="thumb"><img src="' + attr(url) + '" alt=""><button type="button" data-i="' + i + '" aria-label="Remove">×</button></div>';
    }

    /* ---------------- uploads ---------------- */
    function wireDrop(id, accept, multiple, done) {
        var zone = document.getElementById(id);
        var input = document.createElement('input');
        input.type = 'file'; input.accept = accept; input.multiple = multiple; input.style.display = 'none';
        zone.appendChild(input);
        zone.addEventListener('click', function () { input.click(); });
        input.addEventListener('change', function () { if (input.files.length) { done(toArr(input.files)); input.value = ''; } });
        ['dragover', 'dragenter'].forEach(function (ev) { zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add('drag'); }); });
        ['dragleave', 'drop'].forEach(function (ev) { zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove('drag'); }); });
        zone.addEventListener('drop', function (e) { if (e.dataTransfer.files.length) { done(toArr(e.dataTransfer.files)); } });
    }

    function uploadImages(files, bucket, cb) {
        var urls = [];
        var chain = Promise.resolve();
        var zoneMsg = busy();
        files.forEach(function (f) {
            chain = chain.then(function () {
                return uploadOne(f, bucket).then(function (u) { if (u) { urls.push(u); } });
            });
        });
        chain.then(function () { unbusy(zoneMsg); if (urls.length) { cb(urls); toast(urls.length + ' uploaded'); } });
    }
    function uploadStl(file) {
        if (!file) { return; }
        if (!/\.(stl|glb|gltf|obj|3mf)$/i.test(file.name)) {
            toast('Use STL, GLB, glTF, OBJ or 3MF (export from SolidWorks)', true); return;
        }
        var zoneMsg = busy();
        uploadOne(file, BUCKET_STL).then(function (u) { unbusy(zoneMsg); if (u) { ed.stl_url = u; renderStlPill(); toast('Model uploaded'); } });
    }
    function uploadOne(file, bucket) {
        var path = Date.now() + '-' + Math.random().toString(36).slice(2, 7) + '-' + safeName(file.name);
        return window.sb.storage.from(bucket).upload(path, file, { cacheControl: '3600', upsert: false })
            .then(function (res) {
                if (res.error) { toast(res.error.message, true); return null; }
                return window.sb.storage.from(bucket).getPublicUrl(path).data.publicUrl;
            });
    }
    function busy() {
        var n = document.createElement('span');
        n.className = 'uploading'; n.innerHTML = '<span class="mini-spin"></span> Uploading…';
        root.querySelector('.editor-actions').prepend(n);
        return n;
    }
    function unbusy(n) { if (n && n.remove) { n.remove(); } }

    /* ---------------- save ---------------- */
    function saveProject(id) {
        var slug = val('f-slug') || slugify(val('f-title'));
        var specs = toArr(document.querySelectorAll('#spec-rows .spec-row')).map(function (r) {
            return { label: r.querySelector('.spec-label-in').value.trim(), value: r.querySelector('.spec-value-in').value.trim() };
        }).filter(function (s) { return s.label || s.value; });

        var row = {
            slug: slug,
            title: val('f-title').trim(),
            subtitle: val('f-subtitle').trim(),
            status: val('f-status'),
            tags: val('f-tags').split(',').map(function (t) { return t.trim(); }).filter(Boolean),
            summary: val('f-summary').trim(),
            why_md: val('f-why').trim(),
            how_md: val('f-how').trim(),
            problems_md: val('f-problems').trim(),
            hero_image: ed.hero_image || null,
            gallery: ed.gallery,
            stl_url: ed.stl_url || null,
            specs: specs,
            published: document.getElementById('f-pub').checked,
            sort_order: parseInt(val('f-sort'), 10) || 0
        };
        if (!row.title || !row.slug) { toast('Title and slug are required', true); return; }

        var q = id
            ? window.sb.from('projects').update(row).eq('id', id)
            : window.sb.from('projects').insert(row);
        q.then(function (res) {
            if (res.error) { toast(res.error.message, true); return; }
            toast('Saved');
            renderDashboard();
        });
    }

    /* ---------------- export / import ---------------- */
    function exportJson() {
        window.sb.from('projects').select('*').order('status').order('sort_order').then(function (res) {
            if (res.error) { toast(res.error.message, true); return; }
            var clean = (res.data || []).map(function (p) {
                delete p.id; delete p.created_at; delete p.updated_at; return p;
            });
            var blob = new Blob([JSON.stringify(clean, null, 2)], { type: 'application/json' });
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'projects.json';
            a.click();
            URL.revokeObjectURL(a.href);
            toast('Exported — commit it to /data for the offline fallback');
        });
    }

    function importSeed() {
        if (!confirm('Import the starter content from data/projects.json into the database? Existing rows with the same slug will be overwritten.')) { return; }
        fetch('data/projects.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }).then(function (rows) {
            return window.sb.from('projects').upsert(rows, { onConflict: 'slug' });
        }).then(function (res) {
            if (res && res.error) { toast(res.error.message, true); } else { toast('Starter content imported'); loadRows(); }
        }).catch(function (e) { toast('Import failed: ' + e.message, true); });
    }

    /* ---------------- helpers ---------------- */
    function field(label, control, hint) {
        return '<div class="field"><span class="field-label">' + esc(label) + '</span>' + control +
            (hint ? '<span class="hint">' + esc(hint) + '</span>' : '') + '</div>';
    }
    function val(id) { var e = document.getElementById(id); return e ? e.value : ''; }
    function toArr(list) { return Array.prototype.slice.call(list); }
    function slugify(s) {
        return String(s).toLowerCase().trim()
            .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    }
    function safeName(n) { return String(n).replace(/[^a-zA-Z0-9._-]/g, '_'); }
    function fileName(u) { try { return decodeURIComponent(u.split('/').pop()); } catch (e) { return u; } }
    function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
    function attr(s) { return esc(s).replace(/"/g, '&quot;'); }

    var toastTimer;
    function toast(msg, isErr) {
        toastEl.textContent = msg;
        toastEl.className = 'toast show' + (isErr ? ' err' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function () { toastEl.className = 'toast' + (isErr ? ' err' : ''); }, 3500);
    }
})();
