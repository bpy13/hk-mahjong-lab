import { useCallback, useEffect, useRef, useState } from 'react'
import type { BenchmarkProgress, GameView, Hints, Meta, Rules } from './api'
import { api } from './api'
import { benchmarkButtonLabel, benchmarkHeader, isBenchmark } from './benchmark'
import { ActionBar } from './components/ActionBar'
import { GameOver } from './components/GameOver'
import { Hand } from './components/Hand'
import { NewGameDialog } from './components/NewGameDialog'
import type { Position } from './components/Seat'
import { River, SeatEdge } from './components/Seat'
import type { QuizFeedback, StudyMode } from './components/StudyPanel'
import { StudyPanel } from './components/StudyPanel'
import { Tile } from './components/Tile'
import { Wall } from './components/Wall'
import { THEME_LABEL, useTheme, useTileScale } from './theme'
import { PASS, WINDS, isDiscard, relativeSeat, tileName } from './tiles'

const params = new URLSearchParams(window.location.search)
/** Milliseconds between bot moves (`?botDelay=0` for instant play). */
const BOT_DELAY = Number(params.get('botDelay') ?? 600)
/** Solver thinking time for study hints (`?hintMs=`). */
const HINT_MS = Number(params.get('hintMs') ?? 2000)

function loadMode(): StudyMode {
  try {
    const m = localStorage.getItem('studyMode')
    return m === 'hints' || m === 'quiz' ? m : 'off'
  } catch {
    return 'off'
  }
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [view, setView] = useState<GameView | null>(null)
  const [dialog, setDialog] = useState(true)
  const [selected, setSelected] = useState<number | null>(null)
  const [mode, setModeState] = useState<StudyMode>(loadMode)
  const [hints, setHints] = useState<Hints | null>(null)
  const [hintsLoading, setHintsLoading] = useState(false)
  const [showDanger, setShowDanger] = useState(false)
  const [quiz, setQuiz] = useState<QuizFeedback | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showResult, setShowResult] = useState(true)
  const [progress, setProgress] = useState<BenchmarkProgress | null>(null)
  const hintKey = useRef('')
  const [theme, cycleTheme] = useTheme()
  const tableRef = useRef<HTMLElement>(null)
  useTileScale(tableRef, view ? `${view.id}${dialog}` : dialog)

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(String(e)))
  }, [])

  const setMode = (m: StudyMode) => {
    setModeState(m)
    try {
      localStorage.setItem('studyMode', m)
    } catch {
      // storage unavailable: the mode just is not remembered
    }
  }

  const benchmark = isBenchmark(view)
  const finished = !!view?.result

  // The benchmark count: on load, whenever the dialog opens, and after each benchmark hand.
  useEffect(() => {
    if (!dialog && !(benchmark && finished)) return
    api
      .benchmarkProgress()
      .then(setProgress)
      .catch(() => setProgress(null))
  }, [dialog, benchmark, finished])

  const open = async (load: () => Promise<GameView>) => {
    setError(null)
    setDialog(false)
    setHints(null)
    setQuiz(null)
    setSelected(null)
    setShowResult(true)
    try {
      setView(await load())
    } catch (e) {
      setError(String(e))
    }
  }
  const start = (level: string, seat: number | null, rules: Rules) => open(() => api.newGame(level, seat, rules))
  /** The next of Brian's 400 benchmark hands (or the one under way). */
  const startBenchmark = () => open(api.benchmarkNext)

  // Bots move one at a time, paced for readability.
  useEffect(() => {
    if (!view || view.result || !view.bot_to_act) return
    const timer = setTimeout(async () => {
      try {
        setView(await api.step(view.id))
      } catch (e) {
        setError(String(e))
      }
    }, BOT_DELAY)
    return () => clearTimeout(timer)
  }, [view])

  const humanToAct = !!view && !view.result && !view.bot_to_act
  const decisionKey = view ? `${view.id}:${view.events.length}:${view.observation.wall_remaining}:${view.observation.phase}` : ''

  // Study mode: ask the solver about each new decision (never in benchmark hands).
  useEffect(() => {
    if (!view || !humanToAct || benchmark || mode === 'off' || hintKey.current === decisionKey) return
    hintKey.current = decisionKey
    setHints(null)
    setHintsLoading(true)
    api
      .hints(view.id, HINT_MS)
      .then((h) => {
        if (hintKey.current === decisionKey) setHints(h)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setHintsLoading(false))
  }, [view, humanToAct, benchmark, mode, decisionKey])

  const act = useCallback(
    async (action: number) => {
      if (!view || busy) return
      setBusy(true)
      setError(null)
      // In benchmark hands no hints load, so the quiz has nothing to say.
      if (mode === 'quiz' && hints) {
        const label = view.legal.find((l) => l.action === action)?.label ?? ''
        setQuiz({
          chosen: label,
          best: hints.solver.moves[0],
          chosenStats: hints.solver.moves.find((m) => m.action === action) ?? null,
        })
      } else if (mode === 'quiz') {
        setQuiz(null)
      }
      try {
        setView(await api.act(view.id, action))
        setSelected(null)
        setHints(null)
      } catch (e) {
        setError(String(e))
      } finally {
        setBusy(false)
      }
    },
    [view, busy, mode, hints],
  )

  const onSelect = (t: number) => {
    if (selected === t) act(t)
    else setSelected(t)
  }

  // Keys: Enter discards the selected tile, Esc clears, P passes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!view || !humanToAct) return
      if (e.key === 'Enter' && selected !== null && view.legal.some((l) => l.action === selected)) act(selected)
      if (e.key === 'Escape') setSelected(null)
      if ((e.key === 'p' || e.key === 'P') && view.legal.some((l) => l.action === PASS)) act(PASS)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view, humanToAct, selected, act])

  // Esc closes the result window.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowResult(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!meta) return <div className="loading">{error ?? '載入中…'}</div>
  if (dialog || !view)
    return (
      <NewGameDialog
        meta={meta}
        onStart={start}
        onCancel={view ? () => setDialog(false) : undefined}
        onBenchmark={startBenchmark}
        benchmarkLabel={benchmarkButtonLabel(progress)}
        benchmarkDone={progress?.next === null}
      />
    )

  const obs = view.observation
  const botName = view.level in meta.levels ? meta.levels[view.level].label : view.opponents
  const names = (s: number) => (s === view.human ? '你' : `電腦 ${botName}`)
  const canDiscard = humanToAct && view.legal.some((l) => isDiscard(l.action))
  const danger = showDanger && hints ? hints.danger.tiles : undefined
  const claim = obs.claim
  const seatEdge = (s: number, pos: Position) => (
    <SeatEdge
      key={`e${s}`}
      obs={obs}
      seat={s}
      position={pos}
      name={names(s)}
      revealed={view.hands?.[s]}
      delta={view.result?.deltas[s]}
    />
  )
  return (
    <div className="app">
      {error && <div className="error">{error}</div>}
      <main className={benchmark ? 'layout no-study' : 'layout'}>
        <section className="table" ref={tableRef}>
          <header className="table-head">
            <h1>香港麻雀</h1>
            <span className="info">
              {benchmark && view.benchmark_index !== null && (
                <strong data-testid="benchmark-header">{benchmarkHeader(view.benchmark_index)} · </strong>
              )}
              {WINDS[view.rules.prevailing_wind]}圈 · 起糊 {view.rules.min_faan} 番 · 你坐{WINDS[view.human]}
            </span>
            <span className="header-buttons">
              {view.result && !showResult && <button onClick={() => setShowResult(true)}>結果</button>}
              <a className="button" href="#tiles" target="_blank" rel="noreferrer">
                牌樣
              </a>
              <button onClick={cycleTheme} title="切換深淺色">
                {THEME_LABEL[theme]}
              </button>
              <button onClick={() => setDialog(true)}>新局</button>
            </span>
          </header>
          {[0, 1, 2, 3].map((s) => {
            const pos = relativeSeat(s, view.human) as Position
            return pos === 0 ? null : seatEdge(s, pos)
          })}
          {[0, 1, 2, 3].map((s) => (
            <River key={`r${s}`} obs={obs} seat={s} position={relativeSeat(s, view.human) as Position} />
          ))}
          <div className="center">
            <Wall left={obs.wall_remaining} />
            {/* Fixed slots: they keep their space when empty, so nothing moves. */}
            <div className={`claim${claim ? '' : ' empty'}`}>
              {claim ? (
                <>
                  {WINDS[claim[1]]}家{obs.phase === 'rob' ? '槓' : '打出'}
                  <Tile tile={claim[0]} size="sm" highlight />
                </>
              ) : (
                <Tile size="sm" faceDown />
              )}
            </div>
            <div className="muted thinking">
              {!view.result && !humanToAct && obs.current_player !== null ? `${WINDS[obs.current_player]}家思考中…` : '\u00a0'}
            </div>
          </div>
          <div className="mine">
            <div className="mine-row">
              {seatEdge(view.human, 0)}
              <div className="hand-row">
                <Hand
                  hand={obs.hand}
                  lastDrawn={obs.last_drawn}
                  canDiscard={canDiscard}
                  selected={selected}
                  onSelect={onSelect}
                  danger={danger}
                />
                <div className="my-melds">
                  {obs.melds[view.human].map((m, i) => (
                    <span key={i} className="meld">
                      {m.tiles.map((t, j) => (
                        <Tile key={j} tile={t} size="md" faceDown={m.concealed && (j === 0 || j === 3)} />
                      ))}
                    </span>
                  ))}
                </div>
              </div>
            </div>
            <div className="act-row">
              <div className="prompt" data-testid={humanToAct ? 'prompt' : undefined}>
                {!humanToAct && '\u00a0'}
                {humanToAct && obs.phase === 'turn' && '輪到你：揀一隻牌打出'}
                {humanToAct && obs.phase === 'claim' && claim && `${WINDS[claim[1]]}家打出${tileName(claim[0])}：要唔要？`}
                {humanToAct && obs.phase === 'rob' && claim && `${WINDS[claim[1]]}家槓${tileName(claim[0])}：搶槓？`}
              </div>
              <div className="action-slot">
                {humanToAct && <ActionBar legal={view.legal} selected={selected} onAct={act} busy={busy} />}
              </div>
            </div>
          </div>
        </section>
        {/* Benchmark hands measure Brian alone: no study panel (plan Task 12). */}
        {!benchmark && (
          <StudyPanel
            mode={mode}
            setMode={setMode}
            hints={hints}
            loading={hintsLoading}
            showDanger={showDanger}
            setShowDanger={setShowDanger}
            quizFeedback={quiz}
            onPick={(a) => (isDiscard(a) ? setSelected(a) : act(a))}
          />
        )}
      </main>
      {view.result && showResult && (
        <GameOver
          view={view}
          meta={meta}
          onNewGame={() => setDialog(true)}
          onClose={() => setShowResult(false)}
          onNextBenchmark={benchmark && progress?.next !== null ? startBenchmark : undefined}
          benchmarkLabel={benchmark ? benchmarkButtonLabel(progress) : undefined}
        />
      )}
    </div>
  )
}
