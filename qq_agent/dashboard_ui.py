"""Single-page Web Admin Dashboard UI for qq-agent (runtime status, user profiles, L3 memories, L2 summaries, L1 sessions)."""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>qq-agent 后台管理仪表盘</title>
  <style>
    :root { --bg:#0f141c; --panel:#171f2c; --border:#2a364b; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --success:#22c55e; --warn:#f59e0b; --danger:#ef4444; --badge-bg:#1e293b; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.5; padding-bottom:48px; }
    header { background:#111827; border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:50; }
    .brand { display:flex; align-items:center; gap:12px; font-size:18px; font-weight:700; }
    .status-dot { width:10px; height:10px; border-radius:50%; background:var(--success); box-shadow:0 0 8px var(--success); }
    .status-dot.warn { background:var(--warn); box-shadow:0 0 8px var(--warn); }
    .status-dot.off { background:var(--danger); box-shadow:0 0 8px var(--danger); }
    .header-actions { display:flex; gap:10px; align-items:center; }
    .container { max-width:1360px; margin:0 auto; padding:20px 24px; }
    .tabs { display:flex; gap:8px; border-bottom:1px solid var(--border); margin-bottom:20px; flex-wrap:wrap; }
    .tab-btn { background:transparent; color:var(--muted); border:none; padding:10px 18px; font-size:14px; font-weight:600; cursor:pointer; border-bottom:2px solid transparent; transition:all 0.15s; }
    .tab-btn:hover { color:var(--text); }
    .tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
    .tab-pane { display:none; }
    .tab-pane.active { display:block; }
    .grid-4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-bottom:20px; }
    .grid-2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; margin-bottom:20px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }
    .card-title { font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }
    .card-value { font-size:22px; font-weight:700; margin-bottom:6px; }
    .card-sub { font-size:12px; color:var(--muted); }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; background:var(--badge-bg); border:1px solid var(--border); }
    .badge.ok { color:#4ade80; border-color:#166534; background:#052e16; }
    .badge.warn { color:#fbbf24; border-color:#92400e; background:#451a03; }
    .badge.scope { color:#38bdf8; border-color:#0369a1; background:#082f49; }
    .btn { background:var(--accent); color:#0f172a; border:none; border-radius:6px; padding:7px 14px; font-size:13px; font-weight:600; cursor:pointer; }
    .btn:hover { opacity:0.9; }
    .btn-secondary { background:#334155; color:var(--text); }
    .btn-danger { background:#991b1b; color:#fecaca; }
    .btn-sm { padding:4px 10px; font-size:12px; }
    .toolbar { display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .toolbar-group { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    input, select, textarea { background:#0f172a; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:7px 11px; font-size:13px; font-family:inherit; }
    input:focus, select:focus, textarea:focus { outline:none; border-color:var(--accent); }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); vertical-align:top; }
    th { color:var(--muted); font-weight:600; background:#111827; }
    tr:hover td { background:rgba(255,255,255,0.02); }
    .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
    .progress-bar { width:100%; height:8px; background:#0f172a; border-radius:999px; overflow:hidden; margin:6px 0; }
    .progress-fill { height:100%; background:var(--accent); transition:width 0.25s; }
    .chat-box { max-height:540px; overflow-y:auto; display:flex; flex-direction:column; gap:10px; padding:12px; background:#0b0f17; border:1px solid var(--border); border-radius:8px; }
    .msg-item { padding:10px 14px; border-radius:8px; max-width:85%; font-size:13px; white-space:pre-wrap; word-break:break-word; }
    .msg-user { align-self:flex-start; background:#1e293b; border:1px solid #334155; }
    .msg-assistant { align-self:flex-end; background:#082f49; border:1px solid #0369a1; }
    .msg-meta { font-size:11px; color:var(--muted); margin-bottom:4px; }
    #toast { position:fixed; bottom:20px; right:24px; background:#0284c7; color:#fff; padding:10px 18px; border-radius:8px; font-size:13px; font-weight:600; display:none; z-index:100; box-shadow:0 8px 24px rgba(0,0,0,0.4); }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span id="global-dot" class="status-dot"></span>
      <span>qq-agent 后台管理控制台</span>
      <span id="header-persona" class="badge scope">加载中...</span>
    </div>
    <div class="header-actions">
      <span id="last-refresh" class="card-sub"></span>
      <button class="btn btn-secondary btn-sm" onclick="refreshAll()">刷新数据</button>
      <button class="btn btn-sm" onclick="reloadBrain()">无感热重载 Brain</button>
    </div>
  </header>

  <div class="container">
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('overview', this)">总览与运行控制</button>
      <button class="tab-btn" onclick="switchTab('profiles', this)">群友画像管理 (<span id="cnt-profiles">0</span>)</button>
      <button class="tab-btn" onclick="switchTab('memories', this)">L3 长期记忆 (<span id="cnt-memories">0</span>)</button>
      <button class="tab-btn" onclick="switchTab('summaries', this)">L2 前情提要 (<span id="cnt-summaries">0</span>)</button>
      <button class="tab-btn" onclick="switchTab('sessions', this)">L1 实时会话透视 (<span id="cnt-sessions">0</span>)</button>
    </div>

    <div id="tab-overview" class="tab-pane active">
      <div class="grid-4">
        <div class="card">
          <div class="card-title"><span>三进程架构状态</span><span id="rt-badge" class="badge ok">RUNNING</span></div>
          <div class="card-value" id="rt-mode">DOGFOOD-FULL</div>
          <div class="card-sub mono" id="rt-pids">PIDs: -</div>
          <div class="card-sub mono" id="rt-endpoint" style="margin-top:4px;">Endpoint: -</div>
        </div>
        <div class="card">
          <div class="card-title"><span>当前激活人设</span></div>
          <div class="card-value" id="ov-persona-name">-</div>
          <div class="card-sub" id="ov-persona-aliases">唤醒词: -</div>
          <div style="margin-top:10px; display:flex; gap:6px; align-items:center;">
            <select id="persona-select" style="flex:1;"></select>
            <button class="btn btn-sm" onclick="switchPersona()">切换并重载</button>
          </div>
        </div>
        <div class="card">
          <div class="card-title">
            <span>今日调用与 Token 护栏</span>
            <button class="btn btn-secondary btn-sm" onclick="resetQuota()">重置今日额度</button>
          </div>
          <div class="card-value" id="ov-calls">0 / 0 次</div>
          <div class="progress-bar"><div id="ov-calls-bar" class="progress-fill" style="width:0%"></div></div>
          <div class="card-sub" id="ov-tokens">Tokens: 0 / 0 | 成本: $0.00</div>
        </div>
        <div class="card">
          <div class="card-title"><span>群聊激发策略参数</span></div>
          <div class="card-value" id="ov-trigger-main">-</div>
          <div class="card-sub" id="ov-trigger-sub">-</div>
          <div class="card-sub" id="ov-groups" style="margin-top:4px;">白名单群: -</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>跨群自身近况 (W5 Bot Self-State)</span>
            <button class="btn btn-danger btn-sm" onclick="saveSelfState('')">清空近况</button>
          </div>
          <p class="card-sub" style="margin-bottom:8px;">注入各群系统提示词的脱敏机器人自身动态（防注入与去隐私已开启）：</p>
          <div style="display:flex; gap:8px;">
            <input id="self-state-input" type="text" style="flex:1;" placeholder="例如：刚聊过：今晚打战舰世界" />
            <button class="btn btn-sm" onclick="saveSelfState(document.getElementById('self-state-input').value)">保存更新</button>
          </div>
          <div class="card-sub" id="self-state-meta" style="margin-top:6px;"></div>
        </div>
        <div class="card">
          <div class="card-title">
            <span>跨群客观资讯缓存 (Shared Facts)</span>
            <button class="btn btn-danger btn-sm" onclick="clearSharedFacts()">清空全部缓存</button>
          </div>
          <div style="max-height:180px; overflow-y:auto;">
            <table>
              <thead><tr><th>检索词</th><th>摘要结果</th><th>操作</th></tr></thead>
              <tbody id="shared-facts-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div id="tab-profiles" class="tab-pane">
      <div class="card" style="margin-bottom:16px;">
        <div class="card-title"><span>新增 / 修改群友熟人画像 (L3 User Profile)</span></div>
        <div class="toolbar-group" style="margin-top:8px;">
          <input id="prof-uid" type="number" placeholder="QQ 号 (必填)" style="width:160px;" />
          <input id="prof-name" type="text" placeholder="昵称 / 备注名" style="width:200px;" />
          <input id="prof-traits" type="text" placeholder="画像特征（如：战舰世界玩家；喜欢发奶龙表情包）" style="flex:1; min-width:260px;" />
          <button class="btn" onclick="saveProfile()">保存画像</button>
        </div>
      </div>
      <div class="card">
        <div class="toolbar">
          <div class="toolbar-group">
            <input id="prof-search" type="text" placeholder="搜索 QQ 号 / 昵称 / 特征标签..." style="width:280px;" oninput="renderProfiles()" />
            <label class="card-sub" style="display:flex;align-items:center;gap:4px;cursor:pointer;">
              <input id="prof-only-traits" type="checkbox" onchange="renderProfiles()" /> 仅显示有特征标签的群友
            </label>
          </div>
          <span class="card-sub">点击任一行「编辑」可快速载入上方表单修改</span>
        </div>
        <table>
          <thead><tr><th style="width:130px;">QQ 号</th><th style="width:220px;">群名片 / 昵称</th><th>熟人画像特征 (traits_text)</th><th style="width:160px;">最后活跃</th><th style="width:140px;">操作</th></tr></thead>
          <tbody id="profiles-tbody"></tbody>
        </table>
      </div>
    </div>

    <div id="tab-memories" class="tab-pane">
      <div class="card" style="margin-bottom:16px;">
        <div class="card-title"><span>新增 / 编辑 L3 长期记忆 (Circle Memory)</span></div>
        <div class="toolbar-group" style="margin-top:8px;">
          <input id="mem-id" type="hidden" value="" />
          <select id="mem-scope-preset" onchange="onScopePresetChange()">
            <option value="public">public (全群公开记忆)</option>
            <option value="admin_private">admin_private (仅管理员私聊可见)</option>
            <option value="group:1028505110">group:1028505110 (指定群专属)</option>
            <option value="group:916748723">group:916748723 (指定群专属)</option>
            <option value="custom">自定义 scope...</option>
          </select>
          <input id="mem-scope" type="text" value="public" placeholder="作用域 (如 group:123456)" style="width:180px;" />
          <input id="mem-content" type="text" placeholder="长期记忆事实内容..." style="flex:1; min-width:280px;" />
          <button class="btn" id="mem-submit-btn" onclick="saveMemory()">新增记忆</button>
          <button class="btn btn-secondary" id="mem-cancel-btn" style="display:none;" onclick="resetMemoryForm()">取消编辑</button>
        </div>
      </div>
      <div class="card">
        <div class="toolbar">
          <div class="toolbar-group"><input id="mem-search" type="text" placeholder="搜索记忆内容或 scope..." style="width:260px;" oninput="renderMemories()" /></div>
          <span class="card-sub">严格遵循隐私防火墙：admin_private 绝不注入群聊</span>
        </div>
        <table>
          <thead><tr><th style="width:70px;">ID</th><th style="width:180px;">隐私作用域 (Scope)</th><th>记忆内容 (Content)</th><th style="width:160px;">创建时间</th><th style="width:140px;">操作</th></tr></thead>
          <tbody id="memories-tbody"></tbody>
        </table>
      </div>
    </div>

    <div id="tab-summaries" class="tab-pane">
      <div class="card" style="margin-bottom:16px;">
        <div class="card-title"><span>编辑会话滚动前情提要 (L2 Session Summary)</span></div>
        <div class="toolbar-group" style="margin-top:8px; align-items:flex-start;">
          <input id="sum-key" type="text" placeholder="会话 Key (如 group:1028505110)" style="width:220px;" />
          <textarea id="sum-text" rows="3" placeholder="前情提要内容（留空保存即清空该群摘要）..." style="flex:1;"></textarea>
          <button class="btn" onclick="saveSummary()">保存提要</button>
        </div>
      </div>
      <div class="card">
        <table>
          <thead><tr><th style="width:190px;">会话 Key</th><th>当前滚动前情提要 (summary_text)</th><th style="width:100px;">待压缩缓冲</th><th style="width:160px;">更新时间</th><th style="width:150px;">操作</th></tr></thead>
          <tbody id="summaries-tbody"></tbody>
        </table>
      </div>
    </div>

    <div id="tab-sessions" class="tab-pane">
      <div class="card">
        <div class="toolbar">
          <div class="toolbar-group">
            <span>选择会话窗口：</span>
            <select id="session-select" onchange="loadSessionMessages(this.value)" style="min-width:240px;"></select>
            <button class="btn btn-secondary btn-sm" onclick="loadSessionMessages(document.getElementById('session-select').value)">刷新对话</button>
          </div>
          <button class="btn btn-danger btn-sm" onclick="clearCurrentSession()">清空当前会话 L1 窗口</button>
        </div>
        <div id="session-chat-box" class="chat-box"></div>
      </div>
    </div>
  </div>

  <div id="toast"></div>

  <script>
    let STATE = { overview: {}, profiles: [], memories: [], summaries: [], sessions: [], sharedFacts: [] };
    function showToast(msg, isErr = false) {
      const el = document.getElementById('toast');
      el.textContent = msg; el.style.background = isErr ? '#dc2626' : '#0284c7'; el.style.display = 'block';
      clearTimeout(el._timer); el._timer = setTimeout(() => { el.style.display = 'none'; }, 2600);
    }
    function fmtTime(ts) { return ts ? new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false }) : '-'; }
    function esc(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
    function switchTab(name, btn) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      if (btn) btn.classList.add('active');
    }
    async function api(path, opts = {}) {
      const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || ('HTTP ' + res.status));
      return data;
    }
    async function refreshAll() {
      try {
        const [ov, profs, mems, sums, sess] = await Promise.all([
          api('/api/overview'), api('/api/profiles'), api('/api/memories'), api('/api/summaries'), api('/api/sessions')
        ]);
        STATE.overview = ov; STATE.profiles = profs.items || []; STATE.memories = mems.items || [];
        STATE.summaries = sums.items || []; STATE.sessions = sess.sessions || []; STATE.sharedFacts = ov.shared_facts || [];
        document.getElementById('cnt-profiles').textContent = STATE.profiles.length;
        document.getElementById('cnt-memories').textContent = STATE.memories.length;
        document.getElementById('cnt-summaries').textContent = STATE.summaries.length;
        document.getElementById('cnt-sessions').textContent = STATE.sessions.length;
        document.getElementById('last-refresh').textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN', { hour12: false });
        renderOverview(); renderProfiles(); renderMemories(); renderSummaries(); renderSessionSelect();
      } catch (e) { showToast('加载失败: ' + e.message, true); }
    }
    function renderOverview() {
      const ov = STATE.overview, rt = ov.runtime || {}, p = ov.persona || {}, g = ov.guardrails || {}, tr = ov.trigger || {};
      const isRunning = rt.status === 'running' && rt.brain_alive;
      document.getElementById('global-dot').className = 'status-dot ' + (isRunning ? '' : (rt.status === 'running' ? 'warn' : 'off'));
      const badge = document.getElementById('rt-badge');
      badge.textContent = (rt.status || 'unknown').toUpperCase(); badge.className = 'badge ' + (isRunning ? 'ok' : 'warn');
      document.getElementById('rt-pids').textContent = `Sup:${rt.supervisor_pid||'-'} | Daemon:${rt.daemon_pid||'-'}(${rt.daemon_alive?'活':'停'}) | Adapter:${rt.adapter_pid||'-'}(${rt.adapter_alive?'活':'停'}) | Brain:${rt.brain_pid||'-'}(${rt.brain_alive?'活':'停'})`;
      document.getElementById('rt-endpoint').textContent = rt.plain_endpoint || '-';
      document.getElementById('header-persona').textContent = `当前人设: ${p.name || '-'} (${p.active_profile || 'default'})`;
      document.getElementById('ov-persona-name').textContent = `${p.name || '-'} [${p.active_profile || 'default'}]`;
      document.getElementById('ov-persona-aliases').textContent = '唤醒词: ' + ((p.aliases || []).join('、') || '无');
      document.getElementById('persona-select').innerHTML = (p.available_profiles || []).map(prof => `<option value="${esc(prof)}" ${prof === p.active_profile ? 'selected' : ''}>${esc(prof)}</option>`).join('');
      const callsPct = g.max_daily_calls ? Math.min(100, Math.round((g.calls_used / g.max_daily_calls) * 100)) : 0;
      document.getElementById('ov-calls').textContent = `${g.calls_used || 0} / ${g.max_daily_calls || 0} 次 (${callsPct}%)`;
      document.getElementById('ov-calls-bar').style.width = callsPct + '%';
      document.getElementById('ov-tokens').textContent = `Tokens: ${g.tokens_used || 0} / ${g.max_daily_tokens || 0} | 成本: $${((g.cost_micros_used || 0) / 1_000_000).toFixed(4)}`;
      document.getElementById('ov-trigger-main').textContent = `兴趣 ${Math.round((tr.topic_prob||0)*100)}% | 插话 ${Math.round((tr.proactive_prob||0)*100)}% | 保底 ${tr.proactive_every_n||0}条`;
      document.getElementById('ov-trigger-sub').textContent = `追问窗口: ${tr.followup_window_s||0}s (${tr.followup_max_turns||0}轮) | 冷却: ${tr.proactive_cooldown_s||0}s`;
      const tiersStr = Object.entries(tr.group_tiers || {}).map(([k, v]) => `${k}:${v}`).join(', ');
      document.getElementById('ov-groups').textContent = '白名单群: ' + ((tr.allowed_group_ids && tr.allowed_group_ids.length) ? tr.allowed_group_ids.join(', ') : '全部允许') + (tiersStr ? ` | 分级: ${tiersStr}` : '');
      const ss = ov.self_state || {};
      document.getElementById('self-state-input').value = ss.state_text || '';
      document.getElementById('self-state-meta').textContent = ss.updated_at ? ('最后更新: ' + fmtTime(ss.updated_at)) : '当前无跨群近况';
      const sfBody = document.getElementById('shared-facts-tbody');
      sfBody.innerHTML = !STATE.sharedFacts.length ? '<tr><td colspan="3" class="card-sub">暂无缓存条目</td></tr>' : STATE.sharedFacts.map(f => `<tr><td class="mono">${esc(f.query_text)}</td><td>${esc(f.brief_text)}</td><td><button class="btn btn-danger btn-sm" onclick="deleteSharedFact('${esc(f.topic_key)}')">删除</button></td></tr>`).join('');
    }
    function renderProfiles() {
      const q = (document.getElementById('prof-search').value || '').trim().toLowerCase();
      const onlyTraits = document.getElementById('prof-only-traits').checked;
      const rows = STATE.profiles.filter(p => (!onlyTraits || p.traits_text) && (!q || String(p.user_id).includes(q) || (p.display_name||'').toLowerCase().includes(q) || (p.traits_text||'').toLowerCase().includes(q)));
      const tbody = document.getElementById('profiles-tbody');
      tbody.innerHTML = !rows.length ? '<tr><td colspan="5" class="card-sub">无匹配的群友画像记录</td></tr>' : rows.map(p => `<tr><td class="mono">${p.user_id}</td><td>${esc(p.display_name) || '<span class="card-sub">（未记录昵称）</span>'}</td><td>${p.traits_text ? `<span class="badge scope">${esc(p.traits_text)}</span>` : '<span class="card-sub">-</span>'}</td><td class="card-sub">${fmtTime(p.updated_at)}</td><td><button class="btn btn-secondary btn-sm" onclick="fillProfileForm(${p.user_id})">编辑</button> <button class="btn btn-danger btn-sm" onclick="deleteProfile(${p.user_id})">删除</button></td></tr>`).join('');
    }
    function fillProfileForm(uid) {
      const p = STATE.profiles.find(x => x.user_id === uid); if (!p) return;
      document.getElementById('prof-uid').value = p.user_id; document.getElementById('prof-name').value = p.display_name || ''; document.getElementById('prof-traits').value = p.traits_text || '';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    async function saveProfile() {
      const user_id = Number(document.getElementById('prof-uid').value);
      const display_name = document.getElementById('prof-name').value.trim(), traits_text = document.getElementById('prof-traits').value.trim();
      if (!user_id) return showToast('请填写有效的 QQ 号', true);
      try { await api('/api/profiles', { method: 'POST', body: JSON.stringify({ user_id, display_name, traits_text }) }); showToast('群友画像已保存'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function deleteProfile(uid) {
      if (!confirm(`确定删除 QQ ${uid} 的群友画像记录吗？`)) return;
      try { await api('/api/profiles/' + uid, { method: 'DELETE' }); showToast('已删除群友画像'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    function onScopePresetChange() { const v = document.getElementById('mem-scope-preset').value; if (v !== 'custom') document.getElementById('mem-scope').value = v; }
    function renderMemories() {
      const q = (document.getElementById('mem-search').value || '').trim().toLowerCase();
      const rows = STATE.memories.filter(m => !q || (m.scope||'').toLowerCase().includes(q) || (m.content||'').toLowerCase().includes(q));
      const tbody = document.getElementById('memories-tbody');
      tbody.innerHTML = !rows.length ? '<tr><td colspan="5" class="card-sub">暂无 L3 长期记忆条目</td></tr>' : rows.map(m => `<tr><td class="mono">#${m.id}</td><td><span class="badge ${m.scope === 'admin_private' ? 'warn' : 'scope'}">${esc(m.scope)}</span></td><td>${esc(m.content)}</td><td class="card-sub">${fmtTime(m.created_at)}</td><td><button class="btn btn-secondary btn-sm" onclick="editMemory(${m.id})">编辑</button> <button class="btn btn-danger btn-sm" onclick="deleteMemory(${m.id})">删除</button></td></tr>`).join('');
    }
    function editMemory(id) {
      const m = STATE.memories.find(x => x.id === id); if (!m) return;
      document.getElementById('mem-id').value = m.id; document.getElementById('mem-scope').value = m.scope; document.getElementById('mem-content').value = m.content;
      document.getElementById('mem-submit-btn').textContent = '更新记忆 #' + m.id; document.getElementById('mem-cancel-btn').style.display = 'inline-block';
    }
    function resetMemoryForm() {
      document.getElementById('mem-id').value = ''; document.getElementById('mem-content').value = '';
      document.getElementById('mem-submit-btn').textContent = '新增记忆'; document.getElementById('mem-cancel-btn').style.display = 'none';
    }
    async function saveMemory() {
      const id = document.getElementById('mem-id').value ? Number(document.getElementById('mem-id').value) : null;
      const scope = document.getElementById('mem-scope').value.trim(), content = document.getElementById('mem-content').value.trim();
      if (!scope || !content) return showToast('请填写作用域和记忆内容', true);
      try { await api('/api/memories', { method: 'POST', body: JSON.stringify({ id, scope, content }) }); resetMemoryForm(); showToast('L3 长期记忆已保存'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function deleteMemory(id) {
      if (!confirm(`确定删除长期记忆 #${id} 吗？`)) return;
      try { await api('/api/memories/' + id, { method: 'DELETE' }); showToast('记忆已删除'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    function renderSummaries() {
      const tbody = document.getElementById('summaries-tbody');
      tbody.innerHTML = !STATE.summaries.length ? '<tr><td colspan="5" class="card-sub">暂无会话摘要</td></tr>' : STATE.summaries.map(s => `<tr><td class="mono">${esc(s.session_key)}</td><td>${esc(s.summary_text) || '<span class="card-sub">（空）</span>'}</td><td class="mono">${s.pending_count} 条</td><td class="card-sub">${fmtTime(s.updated_at)}</td><td><button class="btn btn-secondary btn-sm" onclick="fillSummaryForm('${esc(s.session_key)}')">编辑</button> <button class="btn btn-danger btn-sm" onclick="deleteSummary('${esc(s.session_key)}')">清空</button></td></tr>`).join('');
    }
    function fillSummaryForm(key) {
      const s = STATE.summaries.find(x => x.session_key === key); if (!s) return;
      document.getElementById('sum-key').value = s.session_key; document.getElementById('sum-text').value = s.summary_text || '';
    }
    async function saveSummary() {
      const session_key = document.getElementById('sum-key').value.trim(), summary_text = document.getElementById('sum-text').value.trim();
      if (!session_key) return showToast('请填写会话 Key', true);
      try { await api('/api/summaries', { method: 'POST', body: JSON.stringify({ session_key, summary_text }) }); showToast('前情提要已更新'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function deleteSummary(key) {
      if (!confirm(`确定清空会话 ${key} 的滚动前情提要吗？`)) return;
      try { await api('/api/summaries', { method: 'POST', body: JSON.stringify({ session_key: key, summary_text: '' }) }); showToast('前情提要已清空'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    function renderSessionSelect() {
      const sel = document.getElementById('session-select'), prev = sel.value;
      sel.innerHTML = STATE.sessions.map(s => `<option value="${esc(s.session_key)}">${esc(s.session_key)} (${s.count} 条消息)</option>`).join('');
      const target = STATE.sessions.some(s => s.session_key === prev) ? prev : (STATE.sessions[0]?.session_key || '');
      if (target) { sel.value = target; loadSessionMessages(target); } else { document.getElementById('session-chat-box').innerHTML = '<div class="card-sub">暂无活跃会话消息</div>'; }
    }
    async function loadSessionMessages(sessionKey) {
      if (!sessionKey) return;
      try {
        const res = await api('/api/sessions?session_key=' + encodeURIComponent(sessionKey));
        const box = document.getElementById('session-chat-box'), msgs = res.messages || [];
        box.innerHTML = !msgs.length ? '<div class="card-sub">该会话暂无消息</div>' : msgs.map(m => `<div class="msg-item ${m.role === 'assistant' ? 'msg-assistant' : 'msg-user'}"><div class="msg-meta">${esc(m.speaker || m.role)} · ${fmtTime(m.created_at)}</div><div>${esc(m.display_text || m.content)}</div></div>`).join('');
        box.scrollTop = box.scrollHeight;
      } catch (e) { showToast(e.message, true); }
    }
    async function clearCurrentSession() {
      const key = document.getElementById('session-select').value;
      if (!key || !confirm(`确定清空会话 ${key} 的 L1 近期对话窗口吗？`)) return;
      try { await api('/api/sessions?session_key=' + encodeURIComponent(key), { method: 'DELETE' }); showToast('会话窗口已清空'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function saveSelfState(state_text) {
      try { await api('/api/self-state', { method: 'POST', body: JSON.stringify({ state_text }) }); showToast(state_text ? '自身近况已更新' : '自身近况已清空'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function deleteSharedFact(topic_key) {
      try { await api('/api/shared-facts?topic_key=' + encodeURIComponent(topic_key), { method: 'DELETE' }); showToast('缓存条目已删除'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function clearSharedFacts() {
      if (!confirm('确定清空全部跨群客观资讯缓存吗？')) return;
      try { await api('/api/shared-facts', { method: 'DELETE' }); showToast('全部缓存已清空'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function switchPersona() {
      const profile = document.getElementById('persona-select').value; if (!profile) return;
      try { await api('/api/actions/switch-persona', { method: 'POST', body: JSON.stringify({ profile }) }); showToast(`已切换人设为 ${profile} 并触发热重载`); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    async function reloadBrain() {
      try { await api('/api/actions/reload-brain', { method: 'POST', body: '{}' }); showToast('已发送 Brain 无感排空热重载信号'); setTimeout(refreshAll, 1200); } catch (e) { showToast(e.message, true); }
    }
    async function resetQuota() {
      if (!confirm('确定重置今日 LLM 调用次数与 Token 预算统计吗？')) return;
      try { await api('/api/actions/reset-quota', { method: 'POST', body: '{}' }); showToast('今日额度已重置'); await refreshAll(); } catch (e) { showToast(e.message, true); }
    }
    refreshAll(); setInterval(refreshAll, 8000);
  </script>
</body>
</html>
"""

__all__ = ["DASHBOARD_HTML"]
