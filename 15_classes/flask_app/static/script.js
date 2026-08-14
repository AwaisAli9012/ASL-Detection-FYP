// ── UI TRANSLATIONS ───────────────────────────────────────────────────────
const UI_TEXT = {
    english: {
        dir: "ltr",
        liveCamera: "LIVE CAMERA",
        detectedSign: "DETECTED SIGN",
        addBtn: "＋ Add to Sentence",
        shortcuts: "KEYBOARD SHORTCUTS",
        shortcutList: [
            ["ENTER", "Add current sign"],
            ["BKSP", "Remove last word"],
            ["G", "Generate AI sentence"],
            ["1", "Speak Self"],
            ["2", "Speak To"],
            ["SPACE", "Clear all words"],
        ],
        signedWords: "SIGNED WORDS",
        emptyWords: "No words yet — press ENTER to add current sign",
        removeBtn: "⌫ Remove",
        clearBtn: "✕ Clear",
        generateBtn: "⚡ Generate",
        aiSection: "AI INTERPRETATION",
        selfLabel: "SELF — FIRST PERSON",
        toLabel: "TO — DIRECTED AT SOMEONE",
        aiPlaceholder: "Press G or click Generate to interpret signed words",
        speakSelf: "🔊 Speak Self",
        speakTo: "🔊 Speak To",
        replySection: "REPLY (Non-Signer)",
        replyHint: "Type a response — it will be spoken aloud in the selected language",
        replyPlaceholder: "Type reply and press Enter...",
        sendBtn: "Send & Speak",
        history: "CONVERSATION HISTORY",
        noHistory: "No conversation yet",
        export: "⬇ Export",
        supportedSigns: "SUPPORTED SIGNS (15)",
        handDetected: "Hand Detected",
        noHand: "No Hand",
        generating: "Generating interpretation...",
        emotionLabel: "EMOTION DETECTED",
    },
    arabic: {
        dir: "rtl",
        liveCamera: "الكاميرا المباشرة",
        detectedSign: "الإشارة المكتشفة",
        addBtn: "＋ أضف إلى الجملة",
        shortcuts: "اختصارات لوحة المفاتيح",
        shortcutList: [
            ["ENTER", "إضافة الإشارة الحالية"],
            ["BKSP", "حذف آخر كلمة"],
            ["G", "توليد جملة بالذكاء الاصطناعي"],
            ["1", "نطق (أنا)"],
            ["2", "نطق (إلى)"],
            ["SPACE", "مسح الكل"],
        ],
        signedWords: "الكلمات المُشار إليها",
        emptyWords: "لا توجد كلمات بعد — اضغط ENTER لإضافة الإشارة الحالية",
        removeBtn: "⌫ حذف",
        clearBtn: "✕ مسح الكل",
        generateBtn: "⚡ توليد",
        aiSection: "التفسير بالذكاء الاصطناعي",
        selfLabel: "أنا — المتكلم",
        toLabel: "إلى — موجّه لشخص آخر",
        aiPlaceholder: "اضغط G أو انقر توليد لتفسير الإشارات",
        speakSelf: "🔊 نطق (أنا)",
        speakTo: "🔊 نطق (إلى)",
        replySection: "الرد (غير المُشير)",
        replyHint: "اكتب ردًا — سيُنطق بصوت عالٍ باللغة المختارة",
        replyPlaceholder: "اكتب الرد واضغط Enter...",
        sendBtn: "إرسال ونطق",
        history: "سجل المحادثة",
        noHistory: "لا توجد محادثة بعد",
        export: "⬇ تصدير",
        supportedSigns: "الإشارات المدعومة (15)",
        handDetected: "يد مكتشفة",
        noHand: "لا يوجد يد",
        generating: "جارٍ التوليد...",
        emotionLabel: "المشاعر المكتشفة",
    },
    urdu: {
        dir: "rtl",
        liveCamera: "لائیو کیمرہ",
        detectedSign: "پہچانا گیا اشارہ",
        addBtn: "＋ جملے میں شامل کریں",
        shortcuts: "کی بورڈ شارٹ کٹس",
        shortcutList: [
            ["ENTER", "موجودہ اشارہ شامل کریں"],
            ["BKSP", "آخری لفظ ہٹائیں"],
            ["G", "AI جملہ بنائیں"],
            ["1", "خود بولیں"],
            ["2", "دوسرے کو بولیں"],
            ["SPACE", "سب صاف کریں"],
        ],
        signedWords: "اشاروں کے الفاظ",
        emptyWords: "کوئی لفظ نہیں — ENTER دبائیں",
        removeBtn: "⌫ ہٹائیں",
        clearBtn: "✕ صاف کریں",
        generateBtn: "⚡ بنائیں",
        aiSection: "AI تشریح",
        selfLabel: "خود — پہلا شخص",
        toLabel: "دوسرے کو — ہدایت",
        aiPlaceholder: "G دبائیں یا Generate پر کلک کریں",
        speakSelf: "🔊 خود بولیں",
        speakTo: "🔊 دوسرے کو بولیں",
        replySection: "جواب (غیر اشارہ کرنے والا)",
        replyHint: "جواب لکھیں — منتخب زبان میں بولا جائے گا",
        replyPlaceholder: "جواب لکھیں اور Enter دبائیں...",
        sendBtn: "بھیجیں اور بولیں",
        history: "گفتگو کی تاریخ",
        noHistory: "ابھی تک کوئی گفتگو نہیں",
        export: "⬇ برآمد کریں",
        supportedSigns: "معاون اشارے (15)",
        handDetected: "ہاتھ پہچانا گیا",
        noHand: "کوئی ہاتھ نہیں",
        generating: "تشریح بن رہی ہے...",
        emotionLabel: "جذبات کا پتہ چلا",
    }
};

// ── STATE ─────────────────────────────────────────────────────────────────
let selfLine = "";
let toLine = "";
let currentLang = "english";

// ── APPLY UI LANGUAGE ─────────────────────────────────────────────────────
function applyUI(lang) {
    const t = UI_TEXT[lang];
    const rtl = t.dir === "rtl";

    // Page direction
    document.body.style.direction = t.dir;

    // Header labels
    document.getElementById('hand-label').textContent = UI_TEXT[lang].noHand;

    // Camera section
    setText('label-emotion', t.emotionLabel);
    setText('label-live-camera', t.liveCamera);
    setText('label-detected-sign', t.detectedSign);
    document.getElementById('btn-add').innerHTML = `${t.addBtn} <kbd>ENTER</kbd>`;

    // Shortcuts
    setText('label-shortcuts', t.shortcuts);
    const sg = document.getElementById('shortcuts-grid');
    if (sg) {
        sg.innerHTML = t.shortcutList.map(([k, desc]) =>
            `<div class="shortcut-row"><kbd>${k}</kbd><span>${desc}</span></div>`
        ).join('');
    }

    // Signed words
    setText('label-signed-words', t.signedWords);
    document.getElementById('btn-remove').innerHTML = `${t.removeBtn} <kbd>BKSP</kbd>`;
    document.getElementById('btn-clear').innerHTML = `${t.clearBtn} <kbd>SPACE</kbd>`;
    document.getElementById('btn-generate').innerHTML = `${t.generateBtn} <kbd>G</kbd>`;

    // Empty hint
    const wd = document.getElementById('words-display');
    if (wd && wd.querySelector('.empty-hint')) {
        wd.innerHTML = `<span class="empty-hint">${t.emptyWords}</span>`;
    }

    // AI section
    setText('label-ai-section', t.aiSection);
    const aiBody = document.getElementById('ai-body');
    if (aiBody && aiBody.querySelector('.ai-placeholder')) {
        aiBody.innerHTML = `<div class="ai-placeholder">${t.aiPlaceholder}</div>`;
    }
    if (aiBody) {
        aiBody.style.direction = t.dir;
        aiBody.style.textAlign = rtl ? 'right' : 'left';
    }

    // Speak buttons
    document.getElementById('btn-speak-self').innerHTML = `${t.speakSelf} <kbd>1</kbd>`;
    document.getElementById('btn-speak-to').innerHTML = `${t.speakTo} <kbd>2</kbd>`;

    // Reply section
    setText('label-reply', t.replySection);
    setText('reply-hint', t.replyHint);
    document.getElementById('reply-input').placeholder = t.replyPlaceholder;
    setText('btn-send', t.sendBtn);

    // History
    setText('label-history', t.history);
    setText('label-signs', t.supportedSigns);
    setText('export-btn', t.export);

    // History direction
    const hl = document.getElementById('history-list');
    if (hl) { hl.style.direction = t.dir; }
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ── LANGUAGE SWITCHER ─────────────────────────────────────────────────────
async function setLanguage(lang) {
    await fetch('/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: lang })
    });
    currentLang = lang;

    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.id === `lang-${lang}`);
    });

    const badges = { english: 'English', arabic: 'العربية', urdu: 'اردو' };
    document.getElementById('lang-badge').textContent = badges[lang] || lang;

    applyUI(lang);
    clearAI();
}

// ── KEYBOARD SHORTCUTS ────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    const replyFocused = document.activeElement === document.getElementById('reply-input');
    if (replyFocused) { if (e.key === 'Enter') sendReply(); return; }
    switch (e.key) {
        case 'Enter': e.preventDefault(); addWord(); flashButton('btn-add'); break;
        case 'Backspace': e.preventDefault(); removeWord(); flashButton('btn-remove'); break;
        case 'g': case 'G': generateSentence(); flashButton('btn-generate'); break;
        case '1': speakLine('self'); flashButton('btn-speak-self'); break;
        case '2': speakLine('to'); flashButton('btn-speak-to'); break;
        case ' ': e.preventDefault(); clearWords(); flashButton('btn-clear'); break;
    }
});

function flashButton(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.add('btn-flash');
    setTimeout(() => el.classList.remove('btn-flash'), 200);
}

// ── POLL SERVER STATE ─────────────────────────────────────────────────────
async function pollState() {
    try {
        const res = await fetch('/state');
        const data = await res.json();
        const t = UI_TEXT[currentLang];

        document.getElementById('fps-val').textContent = data.fps;
        document.getElementById('ms-val').textContent = data.inference_ms;

        const dot = document.getElementById('hand-dot');
        const label = document.getElementById('hand-label');
        if (data.hand_detected) {
            dot.className = 'stat-dot active';
            label.textContent = t.handDetected;
        } else {
            dot.className = 'stat-dot inactive';
            label.textContent = t.noHand;
        }

        const word = data.current_word || '...';
        const conf = data.current_conf;
        document.getElementById('current-sign').textContent = word;
        document.getElementById('overlay-sign').textContent = word;
        document.getElementById('overlay-conf').textContent = data.hand_detected ? `${conf}%` : '';
        document.getElementById('conf-label').textContent = `${conf}%`;

        const bar = document.getElementById('conf-bar');
        bar.style.width = `${conf}%`;
        bar.style.background = conf >= 75 ? 'var(--green)' : 'var(--blue)';

        document.querySelectorAll('.sign-chip').forEach(chip => {
            chip.classList.toggle('active', chip.textContent.toUpperCase() === word);
        });

        renderWords(data.words);

        // Emotion
        const EMOJI = { happy: "😊", sad: "😢", angry: "😠" };
        const COLORS = { happy: "var(--green)", sad: "var(--blue)", angry: "var(--red)" };
        const emo = data.emotion || "...";
        const emoConf = data.emotion_conf || 0;
        const emoEl = document.getElementById('emotion-name');
        const emoEmoji = document.getElementById('emotion-emoji');
        const emoBar = document.getElementById('emotion-bar');
        const emoLbl = document.getElementById('emotion-conf-label');
        if (emoEl) { emoEl.textContent = emo.toUpperCase(); emoEl.style.color = COLORS[emo] || "var(--text-2)"; }
        if (emoEmoji) { emoEmoji.textContent = EMOJI[emo] || "😐"; }
        if (emoBar) { emoBar.style.width = `${emoConf}%`; emoBar.style.background = COLORS[emo] || "var(--gold)"; }
        if (emoLbl) { emoLbl.textContent = `${emoConf}%`; }

    } catch (e) { console.error('Poll error:', e); }
}

setInterval(pollState, 300);

// ── RENDER WORDS ──────────────────────────────────────────────────────────
function renderWords(words) {
    const el = document.getElementById('words-display');
    const t = UI_TEXT[currentLang];
    if (!words || words.length === 0) {
        el.innerHTML = `<span class="empty-hint">${t.emptyWords}</span>`;
        return;
    }
    el.innerHTML = words.map(w => `<span class="word-tag">${w}</span>`).join('');
}

// ── ADD / REMOVE / CLEAR ──────────────────────────────────────────────────
async function addWord() {
    const res = await fetch('/add_word', { method: 'POST' });
    const data = await res.json();
    renderWords(data.words);
}

async function removeWord() {
    const res = await fetch('/remove_word', { method: 'POST' });
    const data = await res.json();
    renderWords(data.words);
}

async function clearWords() {
    await fetch('/clear', { method: 'POST' });
    renderWords([]);
    clearAI();
}

// ── GENERATE ──────────────────────────────────────────────────────────────
async function generateSentence() {
    const t = UI_TEXT[currentLang];
    const aiBody = document.getElementById('ai-body');
    const speakRow = document.getElementById('speak-row');
    aiBody.innerHTML = `<div class="ai-placeholder"><span class="generating"></span> ${t.generating}</div>`;
    speakRow.style.display = 'none';

    const res = await fetch('/generate', { method: 'POST' });
    const data = await res.json();

    if (!data.sentence) {
        aiBody.innerHTML = `<div class="ai-placeholder">${t.aiPlaceholder}</div>`;
        return;
    }

    const lines = data.sentence.split('\n');
    selfLine = '';
    toLine = '';

    const rtl = t.dir === 'rtl';
    aiBody.style.direction = t.dir;
    aiBody.style.textAlign = rtl ? 'right' : 'left';

    let html = '';
    for (const line of lines) {
        if (line.startsWith('Self:')) {
            selfLine = line.replace('Self:', '').trim();
            html += `
        <div class="ai-section">
          <div class="ai-label self">${t.selfLabel}</div>
          <div class="ai-text">${selfLine}</div>
        </div>`;
        } else if (line.startsWith('To:')) {
            toLine = line.replace('To:', '').trim();
            html += `
        <hr class="ai-divider"/>
        <div class="ai-section">
          <div class="ai-label to">${t.toLabel}</div>
          <div class="ai-text">${toLine}</div>
        </div>`;
        }
    }

    aiBody.innerHTML = html || `<div class="ai-text">${data.sentence}</div>`;
    speakRow.style.display = 'flex';
    loadHistory();
}

function clearAI() {
    const t = UI_TEXT[currentLang];
    document.getElementById('ai-body').innerHTML =
        `<div class="ai-placeholder">${t.aiPlaceholder}</div>`;
    document.getElementById('speak-row').style.display = 'none';
    selfLine = '';
    toLine = '';
}

// ── SPEAK ─────────────────────────────────────────────────────────────────
async function speakLine(type) {
    const text = type === 'self' ? selfLine : toLine;
    if (!text) return;
    await fetch('/speak', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
}

// ── REPLY ─────────────────────────────────────────────────────────────────
async function sendReply() {
    const input = document.getElementById('reply-input');
    const message = input.value.trim();
    if (!message) return;
    await fetch('/reply', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    input.value = '';
    loadHistory();
}

// ── HISTORY ───────────────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const t = UI_TEXT[currentLang];
        const res = await fetch('/history');
        const data = await res.json();
        const el = document.getElementById('history-list');
        if (!data.history || data.history.length === 0) {
            el.innerHTML = `<div class="empty-hint">${t.noHistory}</div>`;
            return;
        }
        const reversed = [...data.history].reverse();
        el.innerHTML = reversed.map(entry => `
      <div class="history-item">
        <div class="history-meta">
          <span class="history-time">${entry.timestamp}</span>
          <span class="history-lang">${entry.language || 'English'}</span>
        </div>
        <div class="history-signs">${entry.signs}</div>
        <div class="history-sentence" style="direction:${t.dir}">${entry.sentence}</div>
      </div>
    `).join('');
    } catch (e) { console.error('History error:', e); }
}

setInterval(loadHistory, 5000);

document.addEventListener('DOMContentLoaded', () => {
    applyUI('english');
    loadHistory();
});