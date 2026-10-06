(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };

  // ---------- tabs ----------
  var tabs = [].slice.call(document.querySelectorAll('[role=tab]'));
  function show(id) {
    tabs.forEach(function (t) { t.setAttribute('aria-selected', String(t.dataset.t === id)); });
    document.querySelectorAll('.panel').forEach(function (p) { p.hidden = p.id !== id; });
    if (id === 'layers') initLayers();
    try { history.replaceState(null, '', '#' + id); } catch (e) {}
  }
  tabs.forEach(function (t) { t.addEventListener('click', function () { show(t.dataset.t); }); });
  document.querySelector('.tabs').addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    var i = tabs.indexOf(document.activeElement); if (i < 0) return;
    var n = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length]; n.focus(); show(n.dataset.t);
  });

  // ---------- the catch: wipe compare ----------
  var region = 'serial', diff = false, pos = 50;
  function setImgs() {
    $('w-before').src = 'img/' + region + '_before.png';
    $('w-after').src = 'img/' + region + '_' + (diff ? 'diff' : 'after') + '.png';
    document.querySelector('.tag.r').textContent = diff ? 'Difference map' : 'After the round trip';
  }
  function setPos(p) {
    pos = Math.max(0, Math.min(100, p));
    $('wipe-top').style.width = pos + '%'; $('handle').style.left = pos + '%';
    $('handle').setAttribute('aria-valuenow', Math.round(pos));
  }
  function syncTopImg() { var w = $('wipe').clientWidth; $('w-before').style.width = w + 'px'; }
  $('w-after').addEventListener('load', syncTopImg); window.addEventListener('resize', syncTopImg);
  var drag = false;
  function at(e) { var r = $('wipe').getBoundingClientRect(); setPos((e.clientX - r.left) / r.width * 100); }
  $('wipe').addEventListener('pointerdown', function (e) { drag = true; $('wipe').setPointerCapture(e.pointerId); at(e); });
  $('wipe').addEventListener('pointermove', function (e) { if (drag) at(e); });
  $('wipe').addEventListener('pointerup', function () { drag = false; });
  $('handle').addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { setPos(pos - 5); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setPos(pos + 5); e.preventDefault(); }
  });
  document.querySelectorAll('.seg button').forEach(function (b) {
    b.addEventListener('click', function () {
      region = b.dataset.r;
      document.querySelectorAll('.seg button').forEach(function (x) { x.setAttribute('aria-checked', String(x === b)); });
      setImgs();
    });
  });
  $('diffsw').addEventListener('change', function (e) { diff = e.target.checked; setImgs(); });
  setImgs(); setPos(50);

  // ---------- results ----------
  var results = null;
  fetch('results.json').then(function (r) { return r.json(); }).then(function (r) {
    results = r; var v = r.vae;
    $('vaestats').innerHTML = [
      [v.part_changed_pct + '%', 'of the part\'s pixels changed'],
      [v.part_changed_over_8_pct + '%', 'changed by more than 8 levels out of 255'],
      [v.max_err + ' / 255', 'largest single-pixel error'],
      [v.psnr_db + ' dB', 'PSNR: a normal score for a working autoencoder, still not exact']
    ].map(function (s) { return '<div><dt>' + s[0] + '</dt><dd>' + s[1] + '</dd></div>'; }).join('');
    document.querySelector('#cktable tbody').innerHTML = r.checks.map(function (c) {
      return '<tr><td>' + esc(c.check) + '</td><td><span class="res' + (c.pass ? '' : ' f') + '">' + (c.pass ? 'Pass' : 'Fail') + '</span></td><td>' + esc(c.detail) + '</td></tr>';
    }).join('');
    var m = r.manifest, names = { '01_part.png': 'Part, original pixels + alpha', '02_mask_binary.png': 'Binary mask', '03_mask_edges.png': 'Edge alpha', '04_shadow.png': 'Shadow', '05_background.png': 'Background', '06_composite.png': 'Composite' };
    var files = Object.keys(m.files).concat(['06_composite.png']);
    $('filelist').innerHTML = files.map(function (f) {
      var sh = m.files[f] ? m.files[f].shape : [m.canvas[1], m.canvas[0], 3];
      return '<li><a href="layers/' + f + '" download>' + esc(names[f] || f) + '</a><span>' + sh[1] + ' x ' + sh[0] + '</span></li>';
    }).join('') + '<li><a href="layers/manifest.json" download>manifest.json</a><span>offsets, order, hashes</span></li>';
  });

  // ---------- layers viewer ----------
  var started = false, imgs = {}, data = {}, vis = { bg: true, shadow: true, part: true };
  function load(src) { return new Promise(function (ok, bad) { var i = new Image(); i.onload = function () { ok(i); }; i.onerror = bad; i.src = src; }); }
  function pixels(img) {
    var c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
    var x = c.getContext('2d'); x.drawImage(img, 0, 0); return { w: c.width, d: x.getImageData(0, 0, c.width, c.height).data };
  }
  function initLayers() {
    if (started) return; started = true;
    Promise.all(['05_background', '04_shadow', '01_part', '06_composite', '03_mask_edges'].map(function (n) { return load('layers/' + n + '.png'); })
      .concat([load('img/part_photo.png')])).then(function (a) {
      imgs = { bg: a[0], shadow: a[1], part: a[2], comp: a[3], edge: a[4], photo: a[5] };
      data = { comp: pixels(a[3]), photo: pixels(a[5]), edge: pixels(a[4]) };
      draw();
    });
  }
  function draw() {
    var c = $('cv'), x = c.getContext('2d'), m = results.manifest;
    x.clearRect(0, 0, c.width, c.height);
    if (vis.bg) x.drawImage(imgs.bg, 0, 0);
    if (vis.shadow) x.drawImage(imgs.shadow, 0, 0);
    if (vis.part) x.drawImage(imgs.part, m.part_offset[0], m.part_offset[1]);
  }
  document.querySelectorAll('.toggles input').forEach(function (i) {
    i.addEventListener('change', function () { vis[i.dataset.l] = i.checked; if (imgs.bg) draw(); });
  });
  function rgbAt(p, x, y) { var k = (y * p.w + x) * 4; return [p.d[k], p.d[k + 1], p.d[k + 2], p.d[k + 3]]; }
  $('cv').addEventListener('mousemove', function (e) {
    if (!data.comp || !results) return;
    var c = $('cv'), r = c.getBoundingClientRect(), m = results.manifest;
    var x = Math.floor((e.clientX - r.left) / r.width * c.width), y = Math.floor((e.clientY - r.top) / r.height * c.height);
    var L = $('loupe'), lx = L.getContext('2d'); L.hidden = false;
    L.style.left = Math.min(r.width - 156, (e.clientX - r.left) + 16) + 'px';
    L.style.top = Math.max(0, (e.clientY - r.top) - 166) + 'px';
    lx.imageSmoothingEnabled = false; lx.clearRect(0, 0, 150, 150);
    lx.drawImage(c, x - 7, y - 7, 15, 15, 0, 0, 150, 150);
    lx.strokeStyle = '#fff'; lx.lineWidth = 2; lx.strokeRect(70, 70, 10, 10);
    var ox = x - m.part_offset[0], oy = y - m.part_offset[1], inPart = ox >= 0 && oy >= 0 && ox < m.part_size[0] && oy < m.part_size[1];
    var comp = rgbAt(data.comp, x, y), html = '';
    if (inPart) {
      var a = rgbAt(data.edge, ox, oy)[0];
      var ph = rgbAt(data.photo, ox + m.source_bbox[0], oy + m.source_bbox[1]);
      var same = comp[0] === ph[0] && comp[1] === ph[1] && comp[2] === ph[2];
      var verdict = a === 255 ? (same ? '<span class="verdict ok">Identical to the photo</span>' : '<span class="verdict">Differs from the photo</span>')
        : a === 0 ? '<span class="verdict mix">See-through: target shows here</span>' : '<span class="verdict mix">Edge pixel, blended at alpha ' + a + '</span>';
      html = '<div class="rgb"><span>Composite</span><span>' + comp.slice(0, 3).join(', ') + '</span><span>Photo</span><span>' + ph.slice(0, 3).join(', ') + '</span><span>Position</span><span>' + x + ', ' + y + '</span></div>' + verdict;
    } else {
      html = '<div class="rgb"><span>Composite</span><span>' + comp.slice(0, 3).join(', ') + '</span><span>Position</span><span>' + x + ', ' + y + '</span></div><span class="verdict mix">Target image</span>';
    }
    $('pix').innerHTML = html;
  });
  $('cv').addEventListener('mouseleave', function () { $('loupe').hidden = true; });

  var h = (location.hash || '').slice(1);
  if (['pipe', 'layers', 'checks', 'plan'].indexOf(h) > -1) show(h);
})();
