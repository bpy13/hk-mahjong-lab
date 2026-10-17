import type { Hints, MoveStats } from '../api'
import { planName, tileName } from '../tiles'

export type StudyMode = 'off' | 'hints' | 'quiz'

interface Props {
  mode: StudyMode
  setMode: (m: StudyMode) => void
  hints: Hints | null
  loading: boolean
  showDanger: boolean
  setShowDanger: (b: boolean) => void
  quizFeedback: QuizFeedback | null
  onPick: (action: number) => void
}

export interface QuizFeedback {
  chosen: string
  best: MoveStats
  chosenStats: MoveStats | null
}

const pct = (x: number) => `${Math.round(x * 100)}%`
const pts = (x: number) => `${x >= 0 ? '+' : ''}${x.toFixed(1)}`

function MoveTable({ moves, onPick }: { moves: MoveStats[]; onPick: (a: number) => void }) {
  return (
    <table className="moves" data-testid="hint-moves">
      <thead>
        <tr>
          <th>著法</th>
          <th>期望分</th>
          <th>勝率</th>
          <th>出銃率</th>
        </tr>
      </thead>
      <tbody>
        {moves.slice(0, 3).map((m, i) => (
          <tr key={m.action} className={i === 0 ? 'best' : ''} onClick={() => onPick(m.action)} title="按此選擇">
            <td>{m.label}</td>
            <td>
              {pts(m.mean)} <span className="ci">±{m.ci95.toFixed(1)}</span>
            </td>
            <td>{pct(m.win_rate)}</td>
            <td>{pct(m.deal_in_rate)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Study mode: 提示 shows the solver's top 3 as you play; 測驗 hides them until
 * you have moved, then grades your move. */
export function StudyPanel({ mode, setMode, hints, loading, showDanger, setShowDanger, quizFeedback, onPick }: Props) {
  return (
    <aside className="study" data-testid="study">
      <div className="study-modes">
        {(['off', 'hints', 'quiz'] as StudyMode[]).map((m) => (
          <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
            {{ off: '關', hints: '提示', quiz: '測驗' }[m]}
          </button>
        ))}
      </div>
      <label className="toggle">
        <input type="checkbox" checked={showDanger} onChange={(e) => setShowDanger(e.target.checked)} /> 顯示危險度
      </label>
      {mode !== 'off' && loading && <p className="muted">電腦思考中…</p>}
      {mode === 'hints' && hints && !loading && (
        <>
          <MoveTable moves={hints.solver.moves} onPick={onPick} />
          <p className="muted small">
            模擬 {hints.solver.samples} 局 · {Math.round(hints.solver.elapsed_ms)} 毫秒。± 為九成五信賴區間（只計抽樣誤差）。
          </p>
          <Progress hints={hints} />
          {hints.discards && <DiscardTable hints={hints} />}
        </>
      )}
      {mode === 'quiz' && quizFeedback && (
        <div className="quiz" data-testid="quiz-feedback">
          {quizFeedback.chosenStats && quizFeedback.chosenStats.action === quizFeedback.best.action ? (
            <p className="good">你的「{quizFeedback.chosen}」與電腦首選相同。</p>
          ) : (
            <p className="bad">
              你選「{quizFeedback.chosen}」；電腦首選「{quizFeedback.best.label}」
              {quizFeedback.chosenStats && <>，期望分相差 {(quizFeedback.best.mean - quizFeedback.chosenStats.mean).toFixed(1)}</>}。
            </p>
          )}
        </div>
      )}
      {mode === 'quiz' && !quizFeedback && !loading && <p className="muted">出牌後顯示評分。</p>}
      <details className="help">
        <summary>點樣睇</summary>
        <ul>
          <li>期望分：以電腦模擬對手打到完局的平均得分。</li>
          <li>勝率：自己食糊的機會；出銃率：打出的牌被人食糊的機會。</li>
          <li>向聽：離聽牌尚欠幾隻牌；番數向聽：離「夠三番」的聽牌尚欠幾隻。</li>
          <li>危險度：按公開資訊估計每隻牌出銃的機會（綠安全，紅危險）。</li>
        </ul>
      </details>
    </aside>
  )
}

function Progress({ hints }: { hints: Hints }) {
  const p = hints.progress
  const show = (n: number) => (n < 0 ? '已糊' : n >= 20 ? '未有夠番路線' : String(n))
  return (
    <div className="progress">
      <span>向聽 {show(p.shanten)}</span>
      <span>番數向聽 {show(p.faan_shanten)}</span>
      {p.target && (
        <span>
          目標：{planName(p.target)}（保底 {p.target.faan} 番）
        </span>
      )}
    </div>
  )
}

function DiscardTable({ hints }: { hints: Hints }) {
  const rows = [...(hints.discards ?? [])].sort(
    (a, b) => a.faan_shanten - b.faan_shanten || b.faan_ukeire - a.faan_ukeire,
  )
  return (
    <table className="discards">
      <thead>
        <tr>
          <th>打</th>
          <th>番數向聽</th>
          <th>有效張</th>
          <th>向聽</th>
          <th>進張</th>
          <th>自摸率</th>
        </tr>
      </thead>
      <tbody>
        {rows.slice(0, 8).map((r) => (
          <tr key={r.tile}>
            <td>{tileName(r.tile)}</td>
            <td>{r.faan_shanten >= 20 ? '–' : r.faan_shanten}</td>
            <td>{r.faan_ukeire}</td>
            <td>{r.shanten}</td>
            <td>{r.ukeire}</td>
            <td>{r.solo ? pct(r.solo.win_rate) : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
