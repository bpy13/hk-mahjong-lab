import { useState } from 'react'
import type { GameView, Meta, ReviewItem } from '../api'
import { api } from '../api'
import { WINDS, tileName } from '../tiles'
import { MeldView } from './Seat'
import { Tile } from './Tile'

interface Props {
  view: GameView
  meta: Meta
  onNewGame: () => void
  onClose: () => void
}

/** The end of a hand: every hand revealed, each win's faan breakdown, payments,
 * and the post-hand review. */
export function GameOver({ view, meta, onNewGame, onClose }: Props) {
  const r = view.result!
  const [review, setReview] = useState<ReviewItem[] | null>(null)
  const [loading, setLoading] = useState(false)
  const loadReview = async () => {
    setLoading(true)
    try {
      setReview((await api.review(view.id)).decisions)
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="backdrop" onClick={onClose}>
    <div className="game-over" data-testid="game-over" role="dialog" aria-label="結果" onClick={(e) => e.stopPropagation()}>
      <button className="close" aria-label="關閉" title="關閉 (Esc)" onClick={onClose}>
        ✕
      </button>
      <h2>{r.wins.length === 0 ? '流局' : '食糊'}</h2>
      {r.wins.map((w) => (
        <div key={w.seat} className="win">
          <div className="win-head">
            {WINDS[w.seat]}家{w.seat === view.human ? '（你）' : ''}
            {w.self_drawn ? '自摸' : `食${WINDS[w.from!]}家${w.robbed ? '（搶槓）' : ''}`}
            {w.win_tile !== null && w.win_tile < 34 && <> {tileName(w.win_tile)}</>}
            <strong> {w.score.faan} 番{w.score.limit ? '（爆棚）' : ''}</strong>
            {w.liable !== null && <span className="badge warn">{WINDS[w.liable]}家包</span>}
          </div>
          <div className="win-hand">
            {view.hands![w.seat].map((t, i) => (
              <Tile key={i} tile={t} size="md" />
            ))}
            {!w.self_drawn && w.win_tile !== null && w.win_tile < 34 && <Tile tile={w.win_tile} size="md" highlight />}
            {view.observation.melds[w.seat].map((m, i) => (
              <MeldView key={i} meld={m} size="md" />
            ))}
          </div>
          <ul className="patterns">
            {w.score.patterns.map(([n, f]) => (
              <li key={n}>
                {meta.patterns[String(n)]} {w.score.limit ? '' : `${f} 番`}
              </li>
            ))}
          </ul>
        </div>
      ))}
      <table className="deltas">
        <tbody>
          <tr>
            {r.deltas.map((d, s) => (
              <td key={s} className={d >= 0 ? 'up' : 'down'}>
                {WINDS[s]}
                {s === view.human ? '（你）' : ''} {d > 0 ? '+' : ''}
                {d}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <div className="buttons">
        <button className="primary" onClick={onNewGame}>再來一局</button>
        <button onClick={onClose}>關閉</button>
        <button onClick={loadReview} disabled={loading || review !== null}>
          {loading ? '檢討中…' : '檢討'}
        </button>
      </div>
      {review && <Review items={review} />}
    </div>
    </div>
  )
}

function Review({ items }: { items: ReviewItem[] }) {
  const costly = items.filter((i) => (i.loss ?? 0) > 0.5)
  const total = items.reduce((s, i) => s + Math.max(0, i.loss ?? 0), 0)
  return (
    <div className="review" data-testid="review">
      <p>
        共 {items.length} 個決定，與電腦首選不同而期望分差逾半分的有 {costly.length} 個（合共 {total.toFixed(1)} 分）。
      </p>
      <ol>
        {costly.map((i) => (
          <li key={i.index}>
            第 {i.index + 1} 手：你「{i.chosen_label}」，電腦首選「{i.best.label}」，相差 {(i.loss ?? 0).toFixed(1)} 分
          </li>
        ))}
      </ol>
    </div>
  )
}
