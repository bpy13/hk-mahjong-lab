import type { Meld, Observation } from '../api'
import { WINDS } from '../tiles'
import { Tile } from './Tile'

interface Props {
  obs: Observation
  seat: number
  human: number
  position: number // 0 bottom, 1 right, 2 top, 3 left
  name: string
  revealed?: number[]
  delta?: number
}

export function MeldView({ meld, size = 'sm' }: { meld: Meld; size?: 'sm' | 'md' }) {
  return (
    <span className="meld">
      {meld.tiles.map((t, i) => (
        <Tile key={i} tile={t} size={size} faceDown={meld.concealed && (i === 0 || i === 3)} />
      ))}
    </span>
  )
}

/** One seat: wind, name, melds, bonus tiles, river, and hidden hand size. */
export function Seat({ obs, seat, human, position, name, revealed, delta }: Props) {
  const river = obs.river.filter(([s]) => s === seat)
  const last = obs.river.length > 0 ? obs.river[obs.river.length - 1] : null
  const acting = obs.current_player === seat
  return (
    <div className={`seat pos-${position}${acting ? ' acting' : ''}`} data-testid={`seat-${seat}`}>
      <div className="seat-head">
        <span className="wind">{WINDS[seat]}</span>
        <span className="seat-name">{name}</span>
        {seat === 0 && <span className="badge">莊</span>}
        {obs.liable[seat] !== null && <span className="badge warn" title="十二張落地：包自摸">包 {WINDS[obs.liable[seat]!]}</span>}
        {delta !== undefined && <span className={`delta ${delta >= 0 ? 'up' : 'down'}`}>{delta > 0 ? '+' : ''}{delta}</span>}
      </div>
      <div className="seat-tiles">
        {seat !== human &&
          (revealed ? (
            <span className="revealed">{revealed.map((t, i) => <Tile key={i} tile={t} size="sm" />)}</span>
          ) : (
            <span className="backs">{Array.from({ length: obs.concealed_counts[seat] }, (_, i) => <Tile key={i} size="sm" faceDown />)}</span>
          ))}
        <span className="melds">{obs.melds[seat].map((m, i) => <MeldView key={i} meld={m} />)}</span>
        {obs.bonus[seat].length > 0 && <span className="bonus">{obs.bonus[seat].map((t) => <Tile key={t} tile={t} size="sm" />)}</span>}
      </div>
      <div className="river" data-testid={`river-${seat}`}>
        {river.map(([, t, claimed], i) => (
          <Tile
            key={i}
            tile={t}
            size="sm"
            dim={claimed}
            highlight={last !== null && i === river.length - 1 && last[0] === seat && !claimed}
          />
        ))}
      </div>
    </div>
  )
}
