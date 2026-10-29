import { useState } from 'react'
import type { Meta, Rules } from '../api'
import { DEFAULT_RULES } from '../api'
import { WINDS } from '../tiles'

interface Props {
  meta: Meta
  onStart: (level: string, seat: number | null, rules: Rules) => void
  onCancel?: () => void
  /** Starts the next of Brian's 400 benchmark hands (fixed wall, seat and bots). */
  onBenchmark: () => void
  benchmarkLabel: string
  benchmarkDone: boolean
}

/** Opponents, seat and every rule variant from docs/rules/hk.md, or a benchmark hand. */
export function NewGameDialog({ meta, onStart, onCancel, onBenchmark, benchmarkLabel, benchmarkDone }: Props) {
  const [level, setLevel] = useState('hard')
  const [seat, setSeat] = useState<number | null>(null)
  const [rules, setRules] = useState<Rules>(DEFAULT_RULES)
  const set = <K extends keyof Rules>(k: K, v: Rules[K]) => setRules({ ...rules, [k]: v })
  return (
    <div className="dialog" role="dialog" aria-label="新局">
      <h2>新局</h2>
      <fieldset>
        <legend>對手</legend>
        {Object.entries(meta.levels).map(([k, l]) => (
          <label key={k}>
            <input type="radio" name="level" checked={level === k} onChange={() => setLevel(k)} /> {l.label}
            <span className="muted small"> {l.about}</span>
          </label>
        ))}
      </fieldset>
      <fieldset>
        <legend>座位</legend>
        <label>
          <input type="radio" name="seat" checked={seat === null} onChange={() => setSeat(null)} /> 隨機
        </label>
        {WINDS.map((w, i) => (
          <label key={i}>
            <input type="radio" name="seat" checked={seat === i} onChange={() => setSeat(i)} /> {w}
            {i === 0 ? '（莊）' : ''}
          </label>
        ))}
      </fieldset>
      <fieldset>
        <legend>規則</legend>
        <label>
          <input type="checkbox" checked={rules.flowers} onChange={(e) => set('flowers', e.target.checked)} /> 有花
        </label>
        <label>
          起糊
          <select value={rules.min_faan} onChange={(e) => set('min_faan', Number(e.target.value))}>
            {[0, 1, 2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>
                {n} 番
              </option>
            ))}
          </select>
        </label>
        <label>
          爆棚
          <select value={rules.faan_cap} onChange={(e) => set('faan_cap', Number(e.target.value))}>
            {[8, 10, 13].map((n) => (
              <option key={n} value={n}>
                {n} 番
              </option>
            ))}
          </select>
        </label>
        <label>
          計分
          <select value={rules.payout} onChange={(e) => set('payout', e.target.value)}>
            <option value="half_spicy">半辣上</option>
            <option value="full_spicy">全辣</option>
          </select>
        </label>
        <label>
          一炮多響
          <select value={rules.multiple_winners} onChange={(e) => set('multiple_winners', e.target.value)}>
            <option value="all">全部食糊</option>
            <option value="head_bump">截糊</option>
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={rules.twelve_tile_liability}
            onChange={(e) => set('twelve_tile_liability', e.target.checked)}
          />{' '}
          十二張落地包自摸
        </label>
        <label>
          圈風
          <select value={rules.prevailing_wind} onChange={(e) => set('prevailing_wind', Number(e.target.value))}>
            {WINDS.map((w, i) => (
              <option key={i} value={i}>
                {w}圈
              </option>
            ))}
          </select>
        </label>
      </fieldset>
      <div className="buttons">
        <button className="primary" onClick={() => onStart(level, seat, rules)}>
          開始
        </button>
        {onCancel && <button onClick={onCancel}>取消</button>}
        <button
          data-testid="benchmark-start"
          onClick={onBenchmark}
          disabled={benchmarkDone}
          title="固定牌牆、座位同對手；冇提示"
        >
          {benchmarkLabel}
        </button>
      </div>
    </div>
  )
}
