import { useState, useRef, useEffect } from 'react'
import Anthropic from '@anthropic-ai/sdk'
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts'

// ── Config ───────────────────────────────────────────────────────────────────
const MODEL = 'claude-haiku-4-5'

function getClient() {
  const apiKey = import.meta.env.VITE_ANTHROPIC_API_KEY
  if (!apiKey) throw new Error('Add VITE_ANTHROPIC_API_KEY to .env.local and restart the dev server.')
  return new Anthropic({ apiKey, dangerouslyAllowBrowser: true })
}

// ── API Helpers ───────────────────────────────────────────────────────────────
async function withRetry(fn, retries = 3) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn()
    } catch (e) {
      const is429 = e?.status === 429 || e?.message?.includes('rate_limit')
      if (is429 && attempt < retries - 1) {
        await new Promise(r => setTimeout(r, (attempt + 1) * 8000))
        continue
      }
      throw e
    }
  }
}

async function runWithWebSearch(system, userMessage, maxTokens = 1000) {
  const client = getClient()
  const tools = [{ type: 'web_search_20250305', name: 'web_search' }]
  let messages = [{ role: 'user', content: userMessage }]

  for (let i = 0; i < 3; i++) {
    const res = await withRetry(() =>
      client.messages.create({ model: MODEL, max_tokens: maxTokens, system, messages, tools })
    )

    if (res.stop_reason === 'end_turn') {
      return res.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
    }
    if (res.stop_reason === 'pause_turn') {
      messages = [...messages, { role: 'assistant', content: res.content }]
      continue
    }
    if (res.stop_reason === 'tool_use') {
      const toolResults = res.content
        .filter(b => b.type === 'tool_use')
        .map(b => ({ type: 'tool_result', tool_use_id: b.id, content: '' }))
      messages = [
        ...messages,
        { role: 'assistant', content: res.content },
        { role: 'user', content: toolResults },
      ]
      continue
    }
    break
  }
  throw new Error('API iteration limit reached. Try again.')
}

async function runSimple(system, userMessage, maxTokens = 1200) {
  const client = getClient()
  const res = await withRetry(() =>
    client.messages.create({
      model: MODEL, max_tokens: maxTokens, system,
      messages: [{ role: 'user', content: userMessage }],
    })
  )
  return res.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
}

function parseJSON(text) {
  // Strip markdown fences
  let clean = text.replace(/```(?:json)?/g, '').trim()
  // Try to extract a JSON object or array from anywhere in the text
  const objMatch = clean.match(/\{[\s\S]*\}/)
  const arrMatch = clean.match(/\[[\s\S]*\]/)
  if (arrMatch && (!objMatch || arrMatch.index < objMatch.index)) {
    return JSON.parse(arrMatch[0])
  }
  if (objMatch) return JSON.parse(objMatch[0])
  return JSON.parse(clean)
}

// ── Domain Functions ──────────────────────────────────────────────────────────
async function researchRole(input) {
  const desc = input.url
    ? `Job URL: ${input.url}`
    : `Company: ${input.company}, Role: ${input.role}`

  const text = await runWithWebSearch(
    'Research assistant. Return a single JSON object only. Start with { end with }. No other text.',
    `Research ${desc}. Return JSON: {"company_name":"","role_title":"","company_summary":"1-2 sentences","role_summary":"1 sentence","key_competencies":["c1","c2","c3"],"company_challenges":["ch1","ch2"],"recent_news":"1 item","interview_style":"mixed|behavioral|case-heavy"}`
  )
  return parseJSON(text)
}

async function generateQuestions(research) {
  const text = await runSimple(
    'MBA interview coach. Return a JSON array only. Start with [ end with ]. No other text.',
    `6 questions for ${research.company_name} - ${research.role_title}. Competencies: ${(research.key_competencies || []).slice(0,3).join(', ')}. Style: ${research.interview_style || 'mixed'}. Mix: 3 behavioral, 2 case/strategic, 1 why-company. JSON: [{"id":1,"type":"behavioral","question":"","what_we_are_testing":"","competency":""}]`,
    1000
  )
  return parseJSON(text)
}

async function generateQuestionsFromIntel(blurb, research) {
  const text = await runSimple(
    'MBA interview coach. Return a JSON array only. Start with [ end with ]. No other text.',
    `Recruiter intel: "${blurb.slice(0, 600)}"\nRole: ${research.role_title} at ${research.company_name}\nGenerate 2-3 targeted interview questions this candidate must prepare for based on the intel above.\nJSON: [{"id":"ri-1","type":"behavioral","question":"","what_we_are_testing":"","competency":""}]`,
    800
  )
  return parseJSON(text)
}

async function getFeedback(question, transcript, research, cultureNotes = '') {
  const hasCulture = cultureNotes.trim().length > 0
  const culturePart = hasCulture
    ? `\nCompany culture/values: "${cultureNotes.slice(0, 400)}"\nAlso include "culture_fit":{"score":7,"alignment":"one sentence on what aligned","gaps":"one sentence on what was missing","coaching_tip":"one actionable tip"}`
    : ''

  const text = await runSimple(
    'You are an MBA interview coach. Return JSON only. Start with { end with }. No text before or after.',
    `Evaluate:\nRole: ${research.role_title} at ${research.company_name}\nQ: ${question.question}\nCompetency: ${question.competency || 'General'}\nAnswer: "${(transcript || '').slice(0, 1500)}"${culturePart}\n\nReturn JSON:\n{"overall_score":7,"star_structure":{"situation":true,"task":true,"action":true,"result":false,"feedback":"..."},"strengths":["...","..."],"improvements":["...","..."],"filler_words":["um"],"specificity_score":6,"role_fit_score":8,"model_answer_outline":"..."}`,
    1500
  )
  return parseJSON(text)
}

// ── UI Primitives ─────────────────────────────────────────────────────────────
function ProgressBar({ current, total }) {
  return (
    <div className="fixed top-0 left-0 right-0 z-50 h-1 bg-[#1e2d5a]">
      <div
        className="h-full bg-[#f0c060] transition-all duration-700"
        style={{ width: `${Math.round((current / total) * 100)}%` }}
      />
    </div>
  )
}

function Card({ children, className = '' }) {
  return (
    <div className={`bg-[#0f1628] border border-[#1e2d5a] rounded-2xl p-6 ${className}`}>
      {children}
    </div>
  )
}

function ScoreBadge({ score }) {
  const color = score >= 7 ? 'text-green-400' : score >= 5 ? 'text-yellow-400' : 'text-red-400'
  return (
    <span className={`font-bold text-2xl ${color}`}>
      {score}<span className="text-sm text-gray-500">/10</span>
    </span>
  )
}

function ErrorToast({ message, onDismiss }) {
  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-red-900/90 border border-red-600 text-red-200 px-6 py-3 rounded-xl text-sm shadow-xl flex items-center gap-3 max-w-lg w-[calc(100%-2rem)]">
      <span className="flex-1">{message}</span>
      <button onClick={onDismiss} className="text-red-400 hover:text-white shrink-0">✕</button>
    </div>
  )
}

// ── Screen: Input ─────────────────────────────────────────────────────────────
function InputScreen({ onSubmit, loading }) {
  const [mode, setMode] = useState('manual')
  const [company, setCompany] = useState('')
  const [role, setRole] = useState('')
  const [url, setUrl] = useState('')

  const canSubmit = !loading && (mode === 'url' ? url.trim() : company.trim() && role.trim())

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(mode === 'url' ? { url } : { company, role })
  }

  const inputCls = "w-full bg-[#070d1a] border border-[#1e2d5a] rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-[#f0c060] transition-colors text-sm"

  return (
    <div className="min-h-screen bg-[#070d1a] flex items-center justify-center px-4">
      <div className="w-full max-w-lg">
        <div className="text-center mb-10">
          <div className="text-[#f0c060] text-xs font-semibold tracking-widest uppercase mb-3">AI Interview Coach</div>
          <h1 className="text-4xl font-bold text-white mb-3">Prepare Smarter.</h1>
          <p className="text-gray-400 text-sm">Tailored coaching for MBA Strategy &amp; Operations interviews.</p>
        </div>

        <Card>
          <div className="flex bg-[#070d1a] rounded-xl p-1 mb-6">
            {['manual', 'url'].map(m => (
              <button key={m} onClick={() => setMode(m)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${mode === m ? 'bg-[#1e2d5a] text-white' : 'text-gray-400 hover:text-white'}`}>
                {m === 'manual' ? 'Company + Role' : 'Job Posting URL'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'manual' ? (
              <>
                <div>
                  <label className="block text-xs text-gray-400 uppercase tracking-wider mb-2">Company</label>
                  <input value={company} onChange={e => setCompany(e.target.value)}
                    placeholder="e.g. Stripe, Procter & Gamble, Airbnb" className={inputCls} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 uppercase tracking-wider mb-2">Role Title</label>
                  <input value={role} onChange={e => setRole(e.target.value)}
                    placeholder="e.g. Strategy & Operations Manager" className={inputCls} />
                </div>
              </>
            ) : (
              <div>
                <label className="block text-xs text-gray-400 uppercase tracking-wider mb-2">Job Posting URL</label>
                <input value={url} onChange={e => setUrl(e.target.value)}
                  placeholder="https://..." className={inputCls} />
              </div>
            )}
            <button type="submit" disabled={!canSubmit}
              className="w-full py-3 mt-2 bg-[#f0c060] hover:bg-[#f0d080] disabled:opacity-40 disabled:cursor-not-allowed text-[#070d1a] font-bold rounded-xl transition-colors">
              {loading ? 'Researching…' : 'Research & Generate Questions →'}
            </button>
          </form>
        </Card>

        <p className="text-center text-xs text-gray-600 mt-6">
          Powered by Claude · 6 questions · Audio + AI feedback
        </p>
      </div>
    </div>
  )
}

// ── Screen: Researching ────────────────────────────────────────────────────────
function ResearchingScreen({ company }) {
  const steps = [
    'Searching company intelligence…',
    'Analyzing role requirements…',
    'Identifying key competencies…',
    'Generating tailored questions…',
  ]
  const [step, setStep] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setStep(s => Math.min(s + 1, steps.length - 1)), 2800)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="min-h-screen bg-[#070d1a] flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 rounded-full border-2 border-[#f0c060] border-t-transparent animate-spin mx-auto mb-8" />
        <h2 className="text-xl font-semibold text-white mb-2">
          {company ? `Researching ${company}…` : 'Researching…'}
        </h2>
        <p className="text-[#f0c060] text-sm">{steps[step]}</p>
      </div>
    </div>
  )
}

// ── Screen: Custom Questions ──────────────────────────────────────────────────
function CustomQuestionsScreen({ research, onContinue, onSkip }) {
  const [cultureNotes, setCultureNotes] = useState('')
  const [customQs, setCustomQs] = useState([''])

  const updateQ = (i, val) => setCustomQs(qs => qs.map((q, idx) => idx === i ? val : q))
  const addQ = () => setCustomQs(qs => [...qs, ''])
  const removeQ = (i) => setCustomQs(qs => qs.filter((_, idx) => idx !== i))

  const handleContinue = () => {
    const valid = customQs.filter(q => q.trim())
    onContinue(cultureNotes, valid)
  }

  const textareaCls = "w-full bg-[#070d1a] border border-[#1e2d5a] rounded-xl px-4 py-3 text-gray-300 placeholder-gray-600 focus:outline-none focus:border-[#f0c060] resize-none transition-colors text-sm"
  const inputCls = "flex-1 bg-[#070d1a] border border-[#1e2d5a] rounded-xl px-4 py-3 text-gray-300 placeholder-gray-600 focus:outline-none focus:border-[#f0c060] transition-colors text-sm"

  return (
    <div className="min-h-screen bg-[#070d1a] px-4 py-12">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <div className="text-[#f0c060] text-xs uppercase tracking-widest font-semibold mb-2">Optional · Custom Questions</div>
          <h2 className="text-2xl font-bold text-white">Add Company-Specific Questions</h2>
          <p className="text-gray-400 text-sm mt-2">
            Paste questions the company gave you + their culture values. Your answers will be scored against both general best practices <em>and</em> what this company specifically looks for.
          </p>
        </div>

        {/* Culture notes */}
        <Card className="mb-5">
          <label className="block text-xs uppercase tracking-wider text-gray-400 font-semibold mb-3">
            Company Culture &amp; Values
          </label>
          <textarea
            value={cultureNotes}
            onChange={e => setCultureNotes(e.target.value)}
            rows={4}
            placeholder="e.g. Amazon Leadership Principles: customer obsession, bias for action, ownership, frugality…&#10;&#10;Or: Fast-moving, data-driven, collaborative. They value people who take initiative and can influence without authority."
            className={textareaCls}
          />
          <p className="text-xs text-gray-600 mt-2">This context will be used to evaluate culture fit in every answer.</p>
        </Card>

        {/* Custom questions */}
        <Card className="mb-6">
          <label className="block text-xs uppercase tracking-wider text-gray-400 font-semibold mb-4">
            Custom Questions <span className="text-gray-600 normal-case font-normal">(paste from interview guide or recruiter email)</span>
          </label>
          <div className="space-y-3">
            {customQs.map((q, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="shrink-0 w-6 h-6 bg-[#1e2d5a] rounded-full flex items-center justify-center text-[#f0c060] text-xs font-bold">
                  {i + 1}
                </div>
                <input
                  value={q}
                  onChange={e => updateQ(i, e.target.value)}
                  placeholder={`Custom question ${i + 1}…`}
                  className={inputCls}
                />
                {customQs.length > 1 && (
                  <button onClick={() => removeQ(i)} className="text-gray-600 hover:text-red-400 text-lg shrink-0">×</button>
                )}
              </div>
            ))}
          </div>
          {customQs.length < 5 && (
            <button onClick={addQ} className="mt-4 text-xs text-[#f0c060] hover:text-[#f0d080] flex items-center gap-1">
              + Add another question
            </button>
          )}
        </Card>

        <div className="flex gap-3">
          <button onClick={onSkip}
            className="flex-1 py-3 border border-[#1e2d5a] hover:border-gray-500 text-gray-400 hover:text-white rounded-xl transition-colors">
            Skip →
          </button>
          <button onClick={handleContinue}
            className="flex-1 py-3 bg-[#f0c060] hover:bg-[#f0d080] text-[#070d1a] font-bold rounded-xl transition-colors">
            {cultureNotes.trim() || customQs.some(q => q.trim()) ? 'Add & Begin Interview →' : 'Begin Interview →'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Screen: Recruiter Intel ───────────────────────────────────────────────────
function RecruiterIntelScreen({ research, onAdd, onSkip }) {
  const [blurb, setBlurb] = useState('')
  const [generatedQs, setGeneratedQs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [genError, setGenError] = useState(null)

  const handleGenerate = async () => {
    if (!blurb.trim()) return
    setLoading(true)
    setGenError(null)
    try {
      const qs = await generateQuestionsFromIntel(blurb, research)
      setGeneratedQs(qs)
    } catch (e) {
      setGenError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const textareaCls = "w-full bg-[#070d1a] border border-[#1e2d5a] rounded-xl px-4 py-3 text-gray-300 placeholder-gray-600 focus:outline-none focus:border-[#f0c060] resize-none transition-colors text-sm"

  return (
    <div className="min-h-screen bg-[#070d1a] px-4 py-12">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <div className="text-[#f0c060] text-xs uppercase tracking-widest font-semibold mb-2">Recruiter Intel</div>
          <h2 className="text-2xl font-bold text-white">What did your recruiter tell you?</h2>
          <p className="text-gray-400 text-sm mt-2">
            Paste anything you learned — interview format, what they look for, themes, company priorities. AI will generate targeted questions from it.
          </p>
        </div>

        <Card className="mb-5">
          <textarea
            value={blurb}
            onChange={e => setBlurb(e.target.value)}
            rows={7}
            placeholder={`e.g. The recruiter said the panel values cross-functional leadership stories. Expect a 30-min case on supply chain efficiency. They're big on STAR but also want to see strategic thinking beyond execution. One interviewer is the VP of Operations who cares a lot about metrics and impact…`}
            className={textareaCls}
          />
          {genError && <p className="text-red-400 text-xs mt-2">{genError}</p>}
        </Card>

        {!generatedQs ? (
          <div className="flex gap-3">
            <button onClick={onSkip}
              className="flex-1 py-3 border border-[#1e2d5a] hover:border-gray-500 text-gray-400 hover:text-white rounded-xl transition-colors">
              Skip →
            </button>
            <button onClick={handleGenerate} disabled={!blurb.trim() || loading}
              className="flex-1 py-3 bg-[#f0c060] hover:bg-[#f0d080] disabled:opacity-40 disabled:cursor-not-allowed text-[#070d1a] font-bold rounded-xl transition-colors">
              {loading ? 'Generating…' : 'Generate Questions →'}
            </button>
          </div>
        ) : (
          <>
            <div className="mb-5">
              <div className="text-xs uppercase tracking-wider text-[#f0c060] font-semibold mb-3">
                Generated from your intel
              </div>
              <div className="space-y-3">
                {generatedQs.map((q, i) => (
                  <Card key={i} className="!p-4 flex items-start gap-3">
                    <div className="shrink-0 w-6 h-6 bg-[#f0c060]/20 rounded-full flex items-center justify-center text-[#f0c060] text-xs font-bold">
                      {i + 1}
                    </div>
                    <div>
                      <p className="text-gray-300 text-sm leading-relaxed">{q.question}</p>
                      <p className="text-gray-600 text-xs mt-1">{q.competency}</p>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
            <div className="flex gap-3">
              <button onClick={() => setGeneratedQs(null)}
                className="flex-1 py-3 border border-[#1e2d5a] hover:border-gray-500 text-gray-400 hover:text-white rounded-xl transition-colors">
                Regenerate
              </button>
              <button onClick={() => onAdd(generatedQs)}
                className="flex-1 py-3 bg-[#f0c060] hover:bg-[#f0d080] text-[#070d1a] font-bold rounded-xl transition-colors">
                Add to Interview →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Screen: Questions Overview ─────────────────────────────────────────────────
function QuestionsOverview({ research, questions, onStart, onAddCustom, onAddRecruiterIntel }) {
  const typeTag = {
    behavioral: 'bg-blue-900/60 text-blue-300',
    case: 'bg-purple-900/60 text-purple-300',
    why_company: 'bg-amber-900/60 text-amber-300',
    custom: 'bg-[#f0c060]/20 text-[#f0c060]',
  }

  return (
    <div className="min-h-screen bg-[#070d1a] px-4 py-12">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <div className="text-[#f0c060] text-xs uppercase tracking-widest font-semibold mb-2">Your Interview Session</div>
          <h2 className="text-3xl font-bold text-white">{research.role_title}</h2>
          <p className="text-gray-400 mt-1">{research.company_name} · {research.interview_style}</p>
        </div>

        <Card className="mb-6">
          <p className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-2">Company Snapshot</p>
          <p className="text-gray-300 text-sm leading-relaxed">{research.company_summary}</p>
          {research.recent_news && (
            <p className="text-gray-500 text-xs mt-3 border-t border-[#1e2d5a] pt-3">📰 {research.recent_news}</p>
          )}
        </Card>

        <div className="space-y-3 mb-8">
          {questions.map((q, i) => (
            <Card key={q.id} className="flex items-start gap-4 !p-4">
              <div className="shrink-0 w-7 h-7 bg-[#1e2d5a] rounded-full flex items-center justify-center text-[#f0c060] text-xs font-bold">
                {i + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${typeTag[q.type] || 'bg-gray-800 text-gray-300'}`}>
                    {q.type.replace('_', ' ')}
                  </span>
                  <span className="text-xs text-gray-500">{q.competency}</span>
                </div>
                <p className="text-gray-300 text-sm">{q.question}</p>
              </div>
            </Card>
          ))}
        </div>

        <div className="space-y-3">
          <div className="flex gap-3">
            <button onClick={onAddRecruiterIntel}
              className="flex-1 py-3 border border-[#1e2d5a] hover:border-[#f0c060]/60 text-gray-400 hover:text-[#f0c060] rounded-xl transition-colors text-sm font-medium">
              💬 Recruiter Intel
            </button>
            <button onClick={onAddCustom}
              className="flex-1 py-3 border border-[#1e2d5a] hover:border-[#f0c060]/60 text-gray-400 hover:text-[#f0c060] rounded-xl transition-colors text-sm font-medium">
              + Company Questions
            </button>
          </div>
          <button onClick={onStart}
            className="w-full py-4 bg-[#f0c060] hover:bg-[#f0d080] text-[#070d1a] font-bold rounded-xl transition-colors">
            Begin Interview →
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Screen: Recording ──────────────────────────────────────────────────────────
function RecordingScreen({ question, questionNum, total, onSubmit, isSubmitting }) {
  const [recState, setRecState] = useState('idle') // idle | recording | done
  const [transcript, setTranscript] = useState('')
  const [timer, setTimer] = useState(0)
  const [permError, setPermError] = useState(false)
  const [textMode, setTextMode] = useState(false)

  const recognitionRef = useRef(null)
  const mediaRef = useRef(null)
  const timerRef = useRef(null)
  const finalRef = useRef('')

  const hasSR = !!(window.SpeechRecognition || window.webkitSpeechRecognition)

  useEffect(() => () => {
    clearInterval(timerRef.current)
    recognitionRef.current?.stop()
    mediaRef.current?.stream?.getTracks().forEach(t => t.stop())
  }, [])

  const startRecording = async () => {
    setTranscript('')
    finalRef.current = ''
    setTimer(0)
    setPermError(false)

    if (hasSR && !textMode) {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      const rec = new SR()
      rec.continuous = true
      rec.interimResults = true
      rec.lang = 'en-US'
      rec.onresult = (e) => {
        let final = '', interim = ''
        for (let i = 0; i < e.results.length; i++) {
          if (e.results[i].isFinal) final += e.results[i][0].transcript + ' '
          else interim += e.results[i][0].transcript
        }
        finalRef.current = final
        setTranscript(final + interim)
      }
      rec.onerror = () => setTextMode(true)
      rec.start()
      recognitionRef.current = rec
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRef.current = new MediaRecorder(stream)
      mediaRef.current.start()
      setRecState('recording')
      timerRef.current = setInterval(() => setTimer(t => t + 1), 1000)
    } catch {
      recognitionRef.current?.stop()
      setPermError(true)
    }
  }

  const stopRecording = () => {
    recognitionRef.current?.stop()
    mediaRef.current?.stream?.getTracks().forEach(t => t.stop())
    clearInterval(timerRef.current)
    setRecState('done')
  }

  const reset = () => {
    setRecState('idle')
    setTranscript('')
    setTimer(0)
    finalRef.current = ''
  }

  const fmt = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
  const finalText = (finalRef.current || transcript).trim()

  return (
    <div className="min-h-screen bg-[#070d1a] px-4 py-12">
      <ProgressBar current={questionNum} total={total} />
      <div className="max-w-2xl mx-auto pt-4">

        <div className="flex items-center justify-between mb-6">
          <span className="text-[#f0c060] text-xs uppercase tracking-widest font-semibold">
            Question {questionNum} of {total}
          </span>
          <span className="text-xs text-gray-500 px-3 py-1 bg-[#0f1628] border border-[#1e2d5a] rounded-full capitalize">
            {question.type.replace('_', ' ')} · {question.competency}
          </span>
        </div>

        <Card className="mb-6">
          <p className="text-xl text-white font-medium leading-relaxed">{question.question}</p>
        </Card>

        {permError && (
          <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 mb-4 text-red-300 text-sm">
            Microphone access denied. Allow mic in your browser settings, then reload.
          </div>
        )}

        {!hasSR && !textMode && (
          <div className="bg-amber-900/20 border border-amber-800 rounded-xl p-4 mb-4 text-amber-300 text-sm">
            Live transcription isn't supported in this browser.{' '}
            <button onClick={() => setTextMode(true)} className="underline hover:text-amber-200">
              Type your answer instead.
            </button>
          </div>
        )}

        {/* Mic controls */}
        {!textMode && (
          <div className="flex flex-col items-center gap-4 mb-6">
            {recState === 'idle' && (
              <button onClick={startRecording}
                className="w-20 h-20 bg-[#f0c060] hover:bg-[#f0d080] rounded-full flex items-center justify-center transition-all shadow-lg shadow-yellow-900/20">
                <svg className="w-8 h-8 text-[#070d1a]" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3zm6 9a1 1 0 0 1 2 0 8 8 0 0 1-7 7.93V21h2a1 1 0 0 1 0 2H9a1 1 0 0 1 0-2h2v-2.07A8 8 0 0 1 4 11a1 1 0 0 1 2 0 6 6 0 0 0 12 0z" />
                </svg>
              </button>
            )}
            {recState === 'recording' && (
              <div className="flex flex-col items-center gap-6">
                <div className="relative" onClick={stopRecording}>
                  <div className="w-20 h-20 bg-red-600 rounded-full flex items-center justify-center cursor-pointer animate-pulse">
                    <div className="w-6 h-6 bg-white rounded-sm" />
                  </div>
                  <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-red-400 font-mono text-lg font-bold">
                    {fmt(timer)}
                  </div>
                </div>
                <p className="text-gray-500 text-xs mt-6">Click to stop</p>
              </div>
            )}
          </div>
        )}

        {/* Transcript / text input */}
        {textMode ? (
          <Card className="mb-6">
            <label className="block text-xs uppercase tracking-wider text-gray-400 font-semibold mb-3">
              Your Answer
            </label>
            <textarea
              value={transcript}
              onChange={e => setTranscript(e.target.value)}
              rows={6}
              placeholder="Type your answer here…"
              className="w-full bg-[#070d1a] border border-[#1e2d5a] rounded-xl px-4 py-3 text-gray-300 placeholder-gray-600 focus:outline-none focus:border-[#f0c060] resize-none transition-colors text-sm"
            />
          </Card>
        ) : (recState === 'recording' || recState === 'done') && (
          <Card className="mb-6">
            <div className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-3">
              {recState === 'recording' ? '🔴 Live Transcript' : 'Your Answer'}
            </div>
            <p className="text-gray-300 text-sm leading-relaxed min-h-[60px]">
              {transcript || <span className="text-gray-600 italic">Listening…</span>}
            </p>
          </Card>
        )}

        {/* Actions */}
        {(recState === 'done' || textMode) && !isSubmitting && (
          <div className="flex gap-3">
            <button onClick={reset}
              className="flex-1 py-3 border border-[#1e2d5a] hover:border-gray-500 text-gray-400 hover:text-white rounded-xl transition-colors">
              {textMode ? 'Clear' : 'Re-record'}
            </button>
            <button
              onClick={() => onSubmit(finalText)}
              disabled={!finalText}
              className="flex-1 py-3 bg-[#f0c060] hover:bg-[#f0d080] disabled:opacity-40 disabled:cursor-not-allowed text-[#070d1a] font-bold rounded-xl transition-colors">
              Get Feedback →
            </button>
          </div>
        )}

        {isSubmitting && (
          <div className="flex items-center justify-center gap-3 py-6">
            <div className="w-5 h-5 rounded-full border-2 border-[#f0c060] border-t-transparent animate-spin" />
            <span className="text-gray-400 text-sm">Analyzing your answer…</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Screen: Feedback ───────────────────────────────────────────────────────────
function FeedbackScreen({ question, feedback, questionNum, total, onNext, onRetry }) {
  const [revealed, setRevealed] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setRevealed(r => {
        if (r >= 5) { clearInterval(id); return r }
        return r + 1
      })
    }, 350)
    return () => clearInterval(id)
  }, [])

  const starKeys = ['situation', 'task', 'action', 'result']

  return (
    <div className="min-h-screen bg-[#070d1a] px-4 py-12">
      <ProgressBar current={questionNum} total={total} />
      <div className="max-w-2xl mx-auto pt-4">
        <div className="text-[#f0c060] text-xs uppercase tracking-widest font-semibold mb-6">
          Feedback · Q{questionNum}
        </div>

        {/* Scores */}
        {revealed >= 1 && (
          <Card className="mb-4 flex items-center gap-6">
            <div className="text-center shrink-0">
              <div className="text-xs text-gray-400 mb-1">Overall</div>
              <ScoreBadge score={feedback.overall_score} />
            </div>
            <div className="flex-1 grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-xs text-gray-400 mb-1">Specificity</div>
                <ScoreBadge score={feedback.specificity_score} />
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400 mb-1">Role Fit</div>
                <ScoreBadge score={feedback.role_fit_score} />
              </div>
            </div>
          </Card>
        )}

        {/* STAR */}
        {revealed >= 2 && (
          <Card className="mb-4">
            <div className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-3">STAR Structure</div>
            <div className="grid grid-cols-4 gap-2 mb-3">
              {starKeys.map(k => (
                <div key={k}
                  className={`flex flex-col items-center p-2 rounded-lg border ${feedback.star_structure[k] ? 'bg-green-900/20 border-green-800' : 'bg-red-900/20 border-red-900'}`}>
                  <span className="text-base">{feedback.star_structure[k] ? '✓' : '✗'}</span>
                  <span className="text-xs capitalize text-gray-300 mt-0.5">{k}</span>
                </div>
              ))}
            </div>
            <p className="text-sm text-gray-300 italic">{feedback.star_structure.feedback}</p>
          </Card>
        )}

        {/* Strengths & Improvements */}
        {revealed >= 3 && (
          <div className="grid grid-cols-2 gap-4 mb-4">
            <Card>
              <div className="text-xs uppercase tracking-wider text-green-400 font-semibold mb-3">Strengths</div>
              <ul className="space-y-2">
                {feedback.strengths.map((s, i) => (
                  <li key={i} className="text-sm text-gray-300 flex gap-2">
                    <span className="text-green-400 shrink-0">+</span>{s}
                  </li>
                ))}
              </ul>
            </Card>
            <Card>
              <div className="text-xs uppercase tracking-wider text-amber-400 font-semibold mb-3">Improve</div>
              <ul className="space-y-2">
                {feedback.improvements.map((s, i) => (
                  <li key={i} className="text-sm text-gray-300 flex gap-2">
                    <span className="text-amber-400 shrink-0">→</span>{s}
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        )}

        {/* Model answer + filler words */}
        {revealed >= 4 && (
          <>
            <Card className="mb-4">
              <div className="text-xs uppercase tracking-wider text-[#f0c060] font-semibold mb-2">
                What's being tested
              </div>
              <p className="text-sm text-gray-300">{question.what_we_are_testing}</p>
              <div className="border-t border-[#1e2d5a] mt-3 pt-3">
                <div className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-2">
                  Great Answer Outline
                </div>
                <p className="text-sm text-gray-300 italic">{feedback.model_answer_outline}</p>
              </div>
            </Card>
            {feedback.filler_words?.length > 0 && (
              <div className="flex items-center gap-2 mb-4 flex-wrap">
                <span className="text-xs uppercase tracking-wider text-gray-500">Filler words:</span>
                {feedback.filler_words.map(w => (
                  <span key={w} className="bg-[#1e2d5a] px-2 py-0.5 rounded text-yellow-400 font-mono text-xs">
                    "{w}"
                  </span>
                ))}
              </div>
            )}
          </>
        )}

        {/* Culture Fit Signal */}
        {revealed >= 5 && feedback.culture_fit && (
          <Card className="mb-4 border-[#f0c060]/30">
            <div className="flex items-center justify-between mb-4">
              <div className="text-xs uppercase tracking-wider text-[#f0c060] font-semibold">Culture Fit Signal</div>
              <ScoreBadge score={feedback.culture_fit.score} />
            </div>
            <div className="space-y-3">
              {feedback.culture_fit.alignment && (
                <div className="flex gap-2">
                  <span className="text-green-400 text-xs uppercase tracking-wider shrink-0 pt-0.5">Aligned</span>
                  <p className="text-sm text-gray-300">{feedback.culture_fit.alignment}</p>
                </div>
              )}
              {feedback.culture_fit.gaps && (
                <div className="flex gap-2">
                  <span className="text-amber-400 text-xs uppercase tracking-wider shrink-0 pt-0.5">Gap</span>
                  <p className="text-sm text-gray-300">{feedback.culture_fit.gaps}</p>
                </div>
              )}
              {feedback.culture_fit.coaching_tip && (
                <div className="border-t border-[#1e2d5a] pt-3 mt-1">
                  <p className="text-sm text-[#f0c060] italic">{feedback.culture_fit.coaching_tip}</p>
                </div>
              )}
            </div>
          </Card>
        )}

        <div className="flex gap-3 mt-6">
          <button onClick={onRetry}
            className="flex-1 py-3 border border-[#1e2d5a] hover:border-gray-500 text-gray-400 hover:text-white rounded-xl transition-colors">
            Try Again
          </button>
          <button onClick={onNext}
            className="flex-1 py-3 bg-[#f0c060] hover:bg-[#f0d080] text-[#070d1a] font-bold rounded-xl transition-colors">
            {questionNum < total ? 'Next Question →' : 'See Summary →'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Screen: Summary ────────────────────────────────────────────────────────────
function SummaryScreen({ research, questions, feedbacks }) {
  const avg = arr => arr.length
    ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length * 10) / 10
    : 0

  const radarData = [
    {
      subject: 'Structure',
      value: avg(feedbacks.map(f =>
        f.star_structure
          ? (['situation', 'task', 'action', 'result'].filter(k => f.star_structure[k]).length / 4) * 10
          : 5
      )),
    },
    { subject: 'Specificity', value: avg(feedbacks.map(f => f.specificity_score)) },
    { subject: 'Role Fit', value: avg(feedbacks.map(f => f.role_fit_score)) },
    { subject: 'Content', value: avg(feedbacks.map(f => f.overall_score)) },
    { subject: 'Confidence', value: avg(feedbacks.map(f => Math.max(1, 10 - (f.filler_words?.length ?? 0)))) },
  ]

  const overallAvg = avg(feedbacks.map(f => f.overall_score))
  const verdict = overallAvg >= 7.5
    ? { text: 'Ready to interview', color: 'text-green-400' }
    : overallAvg >= 5.5
      ? { text: 'One more practice round', color: 'text-yellow-400' }
      : { text: 'More prep needed', color: 'text-red-400' }

  const allStrengths = feedbacks.flatMap(f => f.strengths || [])
  const allImprovements = feedbacks.flatMap(f => f.improvements || [])

  return (
    <div className="min-h-screen bg-[#070d1a] px-4 py-12">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-10">
          <div className="text-[#f0c060] text-xs uppercase tracking-widest font-semibold mb-2">Session Complete</div>
          <h2 className="text-3xl font-bold text-white mb-1">{research.company_name}</h2>
          <p className="text-gray-400 text-sm mb-4">{research.role_title}</p>
          <p className={`text-xl font-bold ${verdict.color}`}>{verdict.text}</p>
        </div>

        {/* Radar chart */}
        <Card className="mb-6">
          <div className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-4 text-center">
            Performance Radar
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1e2d5a" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Radar dataKey="value" stroke="#f0c060" fill="#f0c060" fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
        </Card>

        {/* Per-question scores */}
        <Card className="mb-6">
          <div className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-4">
            Question-by-Question
          </div>
          <div className="space-y-3">
            {questions.map((q, i) => (
              <div key={q.id} className="flex items-center gap-3">
                <div className="w-6 h-6 bg-[#1e2d5a] rounded-full flex items-center justify-center text-xs text-[#f0c060] font-bold shrink-0">
                  {i + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-300 truncate mb-1">{q.question}</p>
                  <div className="w-full bg-[#1e2d5a] h-1.5 rounded-full">
                    <div
                      className="h-full bg-[#f0c060] rounded-full transition-all"
                      style={{ width: `${(feedbacks[i]?.overall_score ?? 0) * 10}%` }}
                    />
                  </div>
                </div>
                <span className="text-sm font-bold text-white shrink-0">
                  {feedbacks[i]?.overall_score ?? '–'}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Strengths & focus areas */}
        <div className="grid grid-cols-2 gap-4 mb-8">
          <Card>
            <div className="text-xs uppercase tracking-wider text-green-400 font-semibold mb-3">Top Strengths</div>
            <ul className="space-y-2">
              {allStrengths.slice(0, 3).map((s, i) => (
                <li key={i} className="text-sm text-gray-300 flex gap-2">
                  <span className="text-green-400 shrink-0">+</span>{s}
                </li>
              ))}
            </ul>
          </Card>
          <Card>
            <div className="text-xs uppercase tracking-wider text-amber-400 font-semibold mb-3">Focus Areas</div>
            <ul className="space-y-2">
              {allImprovements.slice(0, 3).map((s, i) => (
                <li key={i} className="text-sm text-gray-300 flex gap-2">
                  <span className="text-amber-400 shrink-0">→</span>{s}
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <button
          onClick={() => window.location.reload()}
          className="w-full py-3 border border-[#1e2d5a] hover:border-[#f0c060] text-gray-400 hover:text-[#f0c060] rounded-xl transition-colors">
          Start New Session
        </button>
      </div>
    </div>
  )
}

// ── App ────────────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState('input')
  const [roleInput, setRoleInput] = useState(null)
  const [research, setResearch] = useState(null)
  const [questions, setQuestions] = useState([])
  const [currentQ, setCurrentQ] = useState(0)
  const [feedbacks, setFeedbacks] = useState([])
  const [cultureNotes, setCultureNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const handleStart = async (input) => {
    setRoleInput(input)
    setLoading(true)
    setError(null)
    setScreen('researching')
    try {
      const res = await researchRole(input)
      const qs = await generateQuestions(res)
      setResearch(res)
      setQuestions(qs)
      setScreen('questions')
    } catch (e) {
      setError(e.message)
      setScreen('input')
    } finally {
      setLoading(false)
    }
  }

  const handleAnswerSubmit = async (transcript) => {
    setIsSubmitting(true)
    setError(null)
    try {
      const fb = await getFeedback(questions[currentQ], transcript, research, cultureNotes)
      setFeedbacks(prev => [...prev, fb])
      setScreen('feedback')
    } catch (e) {
      setError(e.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleRecruiterAdd = (generatedQs) => {
    const tagged = generatedQs.map((q, i) => ({
      ...q,
      id: `ri-${i + 1}`,
      type: q.type || 'behavioral',
      competency: q.competency || 'Recruiter Intel',
    }))
    setQuestions(prev => [...prev, ...tagged])
    setScreen('questions')
  }

  const handleCustomContinue = (notes, customQs) => {
    if (notes.trim()) setCultureNotes(notes)
    if (customQs.length > 0) {
      const extra = customQs.map((q, i) => ({
        id: `custom-${i + 1}`,
        type: 'custom',
        question: q,
        what_we_are_testing: 'Company-specific fit and alignment with their stated values.',
        competency: 'Culture Fit',
      }))
      setQuestions(prev => [...prev, ...extra])
    }
    setCurrentQ(0)
    setScreen('recording')
  }

  const handleNext = () => {
    if (currentQ + 1 >= questions.length) {
      setScreen('summary')
    } else {
      setCurrentQ(c => c + 1)
      setScreen('recording')
    }
  }

  const handleRetry = () => {
    setFeedbacks(prev => prev.slice(0, -1))
    setScreen('recording')
  }

  return (
    <div className="antialiased">
      {error && <ErrorToast message={error} onDismiss={() => setError(null)} />}

      {screen === 'input' && (
        <InputScreen onSubmit={handleStart} loading={loading} />
      )}
      {screen === 'researching' && (
        <ResearchingScreen company={roleInput?.company} />
      )}
      {screen === 'questions' && research && questions.length > 0 && (
        <QuestionsOverview
          research={research}
          questions={questions}
          onStart={() => { setCurrentQ(0); setScreen('recording') }}
          onAddCustom={() => setScreen('custom')}
          onAddRecruiterIntel={() => setScreen('recruiter')}
        />
      )}
      {screen === 'recruiter' && research && (
        <RecruiterIntelScreen
          research={research}
          onAdd={handleRecruiterAdd}
          onSkip={() => setScreen('questions')}
        />
      )}
      {screen === 'custom' && research && (
        <CustomQuestionsScreen
          research={research}
          onContinue={handleCustomContinue}
          onSkip={() => { setCurrentQ(0); setScreen('recording') }}
        />
      )}
      {screen === 'recording' && questions[currentQ] && (
        <RecordingScreen
          question={questions[currentQ]}
          questionNum={currentQ + 1}
          total={questions.length}
          onSubmit={handleAnswerSubmit}
          isSubmitting={isSubmitting}
        />
      )}
      {screen === 'feedback' && feedbacks[feedbacks.length - 1] && (
        <FeedbackScreen
          question={questions[currentQ]}
          feedback={feedbacks[feedbacks.length - 1]}
          questionNum={currentQ + 1}
          total={questions.length}
          onNext={handleNext}
          onRetry={handleRetry}
        />
      )}
      {screen === 'summary' && research && (
        <SummaryScreen
          research={research}
          questions={questions}
          feedbacks={feedbacks}
        />
      )}
    </div>
  )
}
