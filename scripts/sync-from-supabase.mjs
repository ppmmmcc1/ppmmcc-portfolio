#!/usr/bin/env node
/* =====================================================================
   sync-from-supabase.mjs — "bake" the live Supabase content into the repo.

   The public site is STATIC: projects.html / concepts.html / project.html
   read data/projects.json and files in assets/ — they never call Supabase.
   Supabase is only the authoring tool behind /admin.html.

   After you add or edit projects in the admin, run:

       npm run sync          (or: node scripts/sync-from-supabase.mjs)

   This:
     1. pulls every PUBLISHED project from Supabase,
     2. downloads any Supabase-hosted models/images into assets/,
        rewriting their URLs to the committed local paths,
     3. regenerates data/projects.json.

   Then: git add -A && git commit -m "Update projects" && git push

   Config (URL + anon key) is read from js/config.js so there's no
   duplication. Only PUBLISHED rows are returned (anon RLS) — which is
   exactly the set the public site should show.
   ===================================================================== */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const p = (...a) => join(ROOT, ...a);

// ---- read Supabase config from js/config.js (single source of truth) ----
const cfgTxt = readFileSync(p('js', 'config.js'), 'utf8');
const cfg = Function(`const window={};${cfgTxt};return window.SUPA_CONFIG;`)();
if (!cfg || !cfg.url || !cfg.anonKey) {
    console.error('✗ Could not read SUPA_CONFIG from js/config.js'); process.exit(1);
}
const REST = cfg.url.replace(/\/+$/, '') + '/rest/v1';
const headers = { apikey: cfg.anonKey, Authorization: 'Bearer ' + cfg.anonKey };

// columns kept in the committed snapshot (drops id/created_at/updated_at)
const FIELDS = ['slug', 'title', 'subtitle', 'status', 'tags', 'summary', 'why_md',
    'how_md', 'problems_md', 'hero_image', 'gallery', 'stl_url', 'specs',
    'published', 'sort_order'];

// Cloudflare Workers refuses to deploy any single static file over 25 MiB.
const CF_MAX_BYTES = 25 * 1024 * 1024;
const oversized = [];

const isSupa = (u) => typeof u === 'string' && u.includes('supabase.co/storage');
const extOf = (u) => {
    const m = u.split('?')[0].split('#')[0].match(/\.([a-z0-9]{2,5})$/i);
    return m ? m[1].toLowerCase() : 'bin';
};

async function localize(url, relPath) {
    if (!isSupa(url)) { return url; }          // already a local /assets path — leave it
    const res = await fetch(url, { headers });
    if (!res.ok) { throw new Error(`download failed (${res.status}) for ${url}`); }
    const buf = Buffer.from(await res.arrayBuffer());
    mkdirSync(dirname(p(relPath)), { recursive: true });
    writeFileSync(p(relPath), buf);
    console.log(`  ↓ ${relPath}  (${(buf.length / 1024).toFixed(0)} KB)`);
    if (buf.length > CF_MAX_BYTES) {
        oversized.push(`${relPath} (${(buf.length / 1048576).toFixed(1)} MB)`);
    }
    return '/' + relPath.split('\\').join('/');
}

console.log('→ Fetching published projects from Supabase…');
let rows;
try {
    const res = await fetch(`${REST}/projects?select=*&order=status,sort_order`, { headers });
    if (!res.ok) { throw new Error(`HTTP ${res.status}`); }
    rows = await res.json();
} catch (e) {
    console.error(`✗ Supabase read failed: ${e.message}`);
    console.error('  Is the project awake? Open the Supabase dashboard and restore it, then retry.');
    console.error('  (data/projects.json was NOT changed.)');
    process.exit(1);
}

if (!Array.isArray(rows) || rows.length === 0) {
    console.error('✗ Supabase returned 0 published projects — refusing to overwrite data/projects.json.');
    console.error('  Publish at least one project in the admin, then retry.');
    process.exit(1);
}
console.log(`  got ${rows.length} published projects\n→ Localizing Supabase-hosted assets…`);

let pulled = 0;
for (const r of rows) {
    if (isSupa(r.stl_url)) {
        r.stl_url = await localize(r.stl_url, `assets/models/${r.slug}.${extOf(r.stl_url)}`); pulled++;
    }
    if (isSupa(r.hero_image)) {
        r.hero_image = await localize(r.hero_image, `assets/uploads/${r.slug}-hero.${extOf(r.hero_image)}`); pulled++;
    }
    if (Array.isArray(r.gallery)) {
        let i = 0;
        for (const g of r.gallery) {
            i++;
            if (g && isSupa(g.url)) {
                g.url = await localize(g.url, `assets/uploads/${r.slug}-gallery-${i}.${extOf(g.url)}`); pulled++;
            }
        }
    }
}
if (pulled === 0) { console.log('  (nothing new to download — all assets already local)'); }

// projects first (by sort_order), then concepts — keeps the file tidy
const clean = rows
    .map((r) => Object.fromEntries(FIELDS.map((k) => [k, r[k]])))
    .sort((a, b) =>
        (a.status === 'project' ? 0 : 1) - (b.status === 'project' ? 0 : 1) ||
        (a.sort_order || 0) - (b.sort_order || 0) ||
        a.slug.localeCompare(b.slug));

writeFileSync(p('data', 'projects.json'), JSON.stringify(clean, null, 2) + '\n');

const withModels = clean.filter((r) => r.stl_url).length;
console.log(`\n✓ Wrote data/projects.json — ${clean.length} projects, ${withModels} with 3D models.`);
if (oversized.length) {
    console.error('\n✗ These files are over Cloudflare\'s 25 MB per-file limit, so the site WILL NOT deploy until they are smaller:');
    oversized.forEach((f) => console.error('    ' + f));
    console.error('  Shrink them first (e.g. decimate the mesh in your slicer/CAD, or compress the video), re-upload in the admin, and sync again.');
    process.exit(1);
}
console.log('  Next:  git add -A && git commit -m "Update projects" && git push');
